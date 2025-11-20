import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load 环境变量
load_dotenv()

# 配置 llm 和 embedding model
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

print(Settings.text_splitter)

# 加载文件
docs = SimpleDirectoryReader(input_files=["/workspace/data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 将文件转成向量存储
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

# 查询，相比 LangChain 封装了好几个步骤：
# 1. 将 query 转成向量
# 2. 根据 query 的向量找到 top K 个最相似的文本
# 3. 将 query 和 文本一起发送给 LLM
# 4. 解析 LLM 的返回结果
print(query_engine.query("文中举了哪些例子?"))