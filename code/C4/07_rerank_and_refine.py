import os
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from pymilvus import MilvusClient,FieldSchema,DataType,CollectionSchema,Collection,connections

# 导入ColBERT重排器需要的模块
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_core.documents import Document
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable
from pydantic import Field


MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "dual_vector_docs"
MAX_SEQ_LEN = 128  # ColBERT 最大序列长度
COLBERT_DIM = 768   # ColBERT 向量维度

class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(text=text)
        return query_emb.tolist()[0]

class DualVectorMilvusRetriever(BaseRetriever):
    collection: Collection = Field(..., description="Milvus 集合")
    ef: BGEM3EmbeddingFunction = Field(..., description="嵌入函数")
    def __init__(self, collection: Collection, ef: BGEM3EmbeddingFunction, **kwargs):
        # 将参数传递给父类初始化，让 Pydantic 处理验证
        super().__init__(collection=collection, ef=ef, **kwargs)

        
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """实现 BaseRetriever 的核心方法"""
        # 第一阶段：使用 ANN 向量进行粗过滤
        query_ann_vector = self.ef([query])["dense"][0]
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_ann_vector.tolist()],
            anns_field="ann_vector",
            param=search_params,
            limit=10,  # 检索更多候选文档用于重排序
            output_fields=["content", "colbert_vector"]
        )
        
        # 构造候选文档
        candidates = []
        for hit in results[0]:
            doc = Document(
                page_content=hit.entity.get('content'),
                metadata={"colbert_vector": hit.entity.get('colbert_vector')}
            )

            candidates.append(doc)
            
        return candidates     


class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = "bert-base-uncased"

        # 加载模型和分词器
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_texts(self, texts):
        """ColBERT文本编码"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """ColBERT相似度计算（MaxSim操作）"""
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 应用文档mask
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 应用查询mask
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最终分数
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序"""
        if len(documents) == 0:
            return documents

        # 编码查询
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 从文档中获取存储的 colbert_vector（平均向量）
        # 使用平均向量进行相似度计算（虽然不是完整的 MaxSim，但仍然有效）
        doc_vectors = []
        
        for doc in documents:
            # 获取存储的 colbert_vector（平均向量，768维）
            colbert_vec = getattr(doc, 'colbert_vector', None) or doc.metadata.get('colbert_vector')
            
            if isinstance(colbert_vec, list):
                doc_vec = torch.tensor(colbert_vec, dtype=torch.float32)
            else:
                doc_vec = colbert_vec
            
            # 确保是 2D tensor: [1, 768]
            if doc_vec.dim() == 1:
                doc_vec = doc_vec.unsqueeze(0)
            
            # 归一化
            doc_vec = F.normalize(doc_vec, p=2, dim=-1)
            
            doc_vectors.append(doc_vec)
        
        # 使用平均向量计算相似度（点积）
        # query_embeddings: [1, seq_len, 768]，取平均得到 [1, 768]
        query_vec = query_embeddings.mean(dim=1)  # [1, 768]
        
        # 计算点积相似度
        scores = []
        for doc_vec in doc_vectors:
            # doc_vec: [1, 768], query_vec: [1, 768]
            score = torch.matmul(query_vec, doc_vec.transpose(-2, -1)).item()
            scores.append(score)

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs

print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)
print("--> 正在初始化编码器和Milvus客户端...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
reranker = ColBERTReranker()
milvus_client = MilvusClient(uri=MILVUS_URI)
print(f"\n--> 正在创建 Collection 'dual_vector_docs'")
if milvus_client.has_collection("dual_vector_docs"):
    milvus_client.drop_collection("dual_vector_docs")
    print(f"已删除已存在的 Collection: 'dual_vector_docs'")
# 创建 Milvus 集合结构
# 注意：由于完整的 token-level embeddings 数据量太大，无法存储在 Milvus 中
# 因此存储平均后的向量（768维），重排时使用平均向量进行相似度计算
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="ann_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"]),      # 用于ANN检索
    FieldSchema(name="colbert_vector", dtype=DataType.FLOAT_VECTOR, dim=COLBERT_DIM)   # 存储平均后的 ColBERT 向量（768维）
]
schema = CollectionSchema(fields=fields, description="双向量文档存储")
collection = Collection(name="dual_vector_docs", schema=schema)

print(f"\n--> 正在为 'dual_vector_docs' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="ann_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
index_params.add_index(
    field_name="colbert_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name="dual_vector_docs", index_params=index_params)
print(f"\n--> 正在为 'dual_vector_docs' 插入数据")
collection.load()

# 1. 加载和处理文档
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"  # 也可以是 "standard_deviation", "interquartile", "gradient"
)
loader = TextLoader("/workspace/data/C4/txt/ai.txt", encoding="utf-8")
documents = loader.load()
docs = text_splitter.split_documents(documents)
doc_texts = [doc.page_content for doc in docs]

# 生成向量嵌入
ann_vectors = ef(doc_texts)["dense"]
colbert_embeddings = reranker.encode_texts(doc_texts)  # [batch_size, seq_len, 768]

# 处理向量格式：确保都是 Python list
# ann_vectors 转换为列表
if hasattr(ann_vectors, 'tolist'):
    ann_vectors = ann_vectors.tolist()
elif hasattr(ann_vectors, '__iter__'):
    ann_vectors = [v.tolist() if hasattr(v, 'tolist') else list(v) for v in ann_vectors]

# 处理 colbert_embeddings：取平均得到文档级向量
# colbert_embeddings 形状: [batch_size, seq_len, 768]
# 取平均后: [batch_size, 768]
colbert_vectors = colbert_embeddings.mean(dim=1).tolist()  # 对序列维度取平均

# 准备插入数据
data_to_insert = [
    {
        "content": doc_texts[i],
        "ann_vector": ann_vectors[i] if isinstance(ann_vectors[i], list) else ann_vectors[i].tolist(),
        "colbert_vector": colbert_vectors[i] if isinstance(colbert_vectors[i], list) else colbert_vectors[i].tolist()
    }
    for i in range(len(doc_texts))
]

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

base_retriever = DualVectorMilvusRetriever(collection,ef)

# 4. 设置LLM压缩器
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.1, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
compressor = LLMChainExtractor.from_llm(llm)

# 5. 使用DocumentCompressorPipeline组装压缩管道
# 流程: ColBERT重排 -> LLM压缩
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 6. 创建最终的压缩检索器
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 7. 执行查询并展示结果
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询: {query}\n")

# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")

# 7.2 使用管道压缩器的最终结果
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
final_results = final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")
