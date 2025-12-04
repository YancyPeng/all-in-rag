import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def read_txt_files_to_documents(directory_path, chunk_size=1000, chunk_overlap=200):
    """
    读取目录中的所有txt文件，分割后封装成Document对象
    
    Args:
        directory_path (str): 包含txt文件的目录路径
        chunk_size (int): 每个文本块的大小（字符数）
        chunk_overlap (int): 文本块之间的重叠字符数
    
    Returns:
        List[Document]: 分割后的Document对象列表
    """
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    all_documents = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 只处理txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 先创建一个包含完整文件内容的Document
                full_doc = Document(
                    page_content=content,
                    metadata={
                        'source': filename,
                        'file_path': file_path,
                        'type': 'full_document'
                    }
                )
                
                # 分割文档
                chunks = text_splitter.split_documents([full_doc])
                
                # 为每个chunk添加额外的metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'type': 'chunk'
                    })
                
                all_documents.extend(chunks)
                print(f"已读取文件: {filename}，分割为 {len(chunks)} 个块")
                
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    return all_documents

def read_txt_files_recursive(directory_path, chunk_size=1000, chunk_overlap=200):
    """
    递归读取目录及其子目录中的所有txt文件，分割后封装成Document对象
    
    Args:
        directory_path (str): 根目录路径
        chunk_size (int): 每个文本块的大小（字符数）
        chunk_overlap (int): 文本块之间的重叠字符数
    
    Returns:
        List[Document]: 分割后的Document对象列表
    """
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    all_documents = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # 计算相对路径作为metadata
                    relative_path = os.path.relpath(file_path, directory_path)
                    
                    # 先创建一个包含完整文件内容的Document
                    full_doc = Document(
                        page_content=content,
                        metadata={
                            'source': filename,
                            'file_path': file_path,
                            'relative_path': relative_path,
                            'type': 'full_document'
                        }
                    )
                    
                    # 分割文档
                    chunks = text_splitter.split_documents([full_doc])
                    
                    # 为每个chunk添加额外的metadata
                    for i, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            'chunk_id': i,
                            'total_chunks': len(chunks),
                            'type': 'chunk'
                        })
                    
                    all_documents.extend(chunks)
                    print(f"已读取文件: {relative_path}，分割为 {len(chunks)} 个块")
                    
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    
    return all_documents

def load_vectorstore(faiss_path: str, embeddings_model_name: str = "BAAI/bge-small-zh-v1.5"):
    """
    加载已保存的FAISS向量存储
    
    Args:
        faiss_path (str): FAISS索引文件路径
        embeddings_model_name (str): 嵌入模型名称
    
    Returns:
        FAISS: 加载的向量存储对象
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def rag_query(vectorstore, query: str, k: int = 3, llm_model: str = "llama2", temperature: float = 0.1) -> Dict[str, Any]:
    """
    RAG查询：根据query检索相关文档，然后使用LLM生成答案
    
    Args:
        vectorstore: FAISS向量存储对象
        query (str): 用户查询
        k (int): 检索的文档数量，默认为3
        llm_model (str): LLM模型名称，默认为"llama2"
        temperature (float): LLM温度参数，默认为0.1
    
    Returns:
        Dict[str, Any]: 包含LLM生成的answer和检索到的contexts
            {
                "answer": "LLM生成的答案",
                "contexts": [检索到的文档列表],
                "query": "原始查询",
                "total_results": 3
            }
    """
    try:
        # 1. 检索相关文档
        search_results = vectorstore.similarity_search_with_score(query, k=k)
        
        # 构建contexts列表
        contexts = []
        for i, (doc, score) in enumerate(search_results):
            context_item = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "similarity": 1 - float(score),
                "rank": i + 1
            }
            contexts.append(context_item)
        
        # 2. 如果没有找到相关文档，返回默认答案
        if not contexts:
            return {
                "answer": "抱歉，我没有找到与您的问题相关的信息。",
                "contexts": [],
                "query": query,
                "total_results": 0
            }
        
        # 3. 准备上下文文本
        context_text = "\n\n".join([
            f"[文档{i+1}] {ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # 4. 创建提示模板
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个有用的AI助手。请根据以下提供的上下文信息来回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请基于上述上下文信息回答问题。如果上下文中没有相关信息，请诚实地说明，不要编造答案。

回答："""
        )
        
        # 5. 初始化LLM
        try:
            llm = ChatDeepSeek(
                model="deepseek-chat",
                temperature=0.7,
                max_tokens=2048,
                api_key=os.getenv("DEEPSEEK_API_KEY")
)
        except Exception as e:
            print(f"无法连接到Ollama服务: {e}")
            # 如果LLM不可用，返回基于检索结果的简单答案
            best_context = contexts[0]
            return {
                "answer": f"根据相关文档，{best_context['content'][:200]}...",
                "contexts": contexts,
                "query": query,
                "total_results": len(contexts),
                "llm_error": str(e)
            }
        
        # 6. 生成答案
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run(context=context_text, question=query)
        
        # 7. 清理答案文本
        answer = answer.strip()
        
        return {
            "answer": answer,
            "contexts": contexts,
            "query": query,
            "total_results": len(contexts),
            "llm_model": llm_model
        }
        
    except Exception as e:
        return {
            "answer": f"查询过程中出现错误: {str(e)}",
            "contexts": contexts if 'contexts' in locals() else [],
            "query": query,
            "total_results": len(contexts) if 'contexts' in locals() else 0,
            "error": str(e)
        }

def similarity_search_with_score(vectorstore, query: str, k: int = 3) -> Dict[str, Any]:
    """
    执行带相似度分数的搜索并返回结构化的结果
    
    Args:
        vectorstore: FAISS向量存储对象
        query (str): 查询字符串
        k (int): 返回的最相似文档数量，默认为3
    
    Returns:
        Dict[str, Any]: 包含answer和contexts的字典，contexts中包含相似度分数
    """
    try:
        # 执行带分数的相似性搜索
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # 构建contexts列表
        contexts = []
        for i, (doc, score) in enumerate(results):
            context_item = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),  # 相似度分数，越小越相似
                "similarity": 1 - float(score),  # 转换为相似度（0-1，越大越相似）
                "rank": i + 1
            }
            contexts.append(context_item)
        
        # 生成包含分数信息的answer
        if contexts:
            best_match = contexts[0]
            answer = f"找到 {len(contexts)} 个相关文档片段，最匹配的内容来自 '{best_match['metadata'].get('source', '未知文件')}'，相似度: {best_match['similarity']:.3f}"
        else:
            answer = "未找到相关文档"
        
        return {
            "answer": answer,
            "contexts": contexts,
            "query": query,
            "total_results": len(contexts)
        }
        
    except Exception as e:
        return {
            "answer": f"搜索出错: {str(e)}",
            "contexts": [],
            "query": query,
            "total_results": 0,
            "error": str(e)
        }

# 示例用法
if __name__ == "__main__":
    # 方法1: 读取单个目录中的txt文件（带分割）
    txt_directory = "/workspace/data/C4/txt"  # 替换为你的txt文件目录
    documents = read_txt_files_to_documents(txt_directory, chunk_size=800, chunk_overlap=150)
    
    print(f"\n总共读取并分割了 {len(documents)} 个文档块")
    
    # 显示分割结果示例
    if documents:
        print("\n=== 分割结果示例 ===")
        for i, doc in enumerate(documents[:3]):  # 显示前3个块
            print(f"\n块 {i+1}:")
            print(f"  来源: {doc.metadata['source']}")
            print(f"  块ID: {doc.metadata['chunk_id']}/{doc.metadata['total_chunks']}")
            print(f"  内容预览: {doc.page_content[:150]}...")
    
    # 方法2: 递归读取整个data目录中的所有txt文件（带分割）
    # data_directory = "./data"  # 替换为你的根目录
    # all_documents = read_txt_files_recursive(data_directory, chunk_size=1000, chunk_overlap=200)
    
    # print(f"\n递归读取总共 {len(all_documents)} 个文档块")
    
    # 如果有文档，创建向量存储
    if documents:
        print("\n正在创建向量存储...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # 保存向量存储
        local_faiss_path = "./faiss_index_store"
        vectorstore.save_local(local_faiss_path)
        print(f"FAISS索引已保存到 {local_faiss_path}")
        
        # 示例查询 - 使用RAG函数
        query = "蜂医是什么？"
        
        # 方法1: 基本搜索（不使用LLM）
        result1 = similarity_search_with_context(vectorstore, query, k=3)
        print(f"\n=== 基本搜索结果 ===")
        print(f"查询: {result1['query']}")
        print(f"回答: {result1['answer']}")
        print(f"结果数量: {result1['total_results']}")
        
        # 方法2: RAG查询（使用LLM生成答案）
        print(f"\n=== RAG查询结果 ===")
        rag_result = rag_query(vectorstore, query, k=3, llm_model="llama2")
        print(f"查询: {rag_result['query']}")
        print(f"LLM答案: {rag_result['answer']}")
        print(f"检索到的文档数量: {rag_result['total_results']}")
        


    
    # 示例：加载已存在的向量存储并进行RAG查询
    print(f"\n=== 加载已存在的向量存储并进行RAG查询 ===")
    try:
        loaded_vectorstore = load_vectorstore("./faiss_index_store")
        test_queries = [
            "支援型干员有什么特点？",
            "蜂医的技能是什么？",
            "激素枪有什么作用？"
        ]
        
        for test_query in test_queries:
            print(f"\n{'='*50}")
            rag_result = rag_query(loaded_vectorstore, test_query, k=2, llm_model="llama2")
            print(f"问题: {test_query}")
            print(f"答案: {rag_result['answer']}")
            
    except Exception as e:
        print(f"RAG查询失败: {e}")
        
    # 使用说明
    print(f"\n=== 使用说明 ===")
    print("1. 确保已安装并运行Ollama服务")
    print("2. 在Ollama中拉取模型: ollama pull llama2")
    print("3. 使用rag_query()函数进行RAG查询")
    print("4. 如果Ollama不可用，函数会返回基于检索结果的简单答案")