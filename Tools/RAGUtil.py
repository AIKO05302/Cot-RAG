from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import SeleniumURLLoader
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(base_url="http://127.0.0.1:11434",model="nomic-embed-text")
vector_store = Chroma(
            collection_name="foo",
            embedding_function=embeddings,
            persist_directory=r"F:\AI\project\Cot-RAG\bak\chroma_db"  # 保存路径
        )
def read_rag_content(
        query_str: str,
) -> str:
    """用于从一个rag系统中读取内容"""
    print("正在从rag中读取内容... " + query_str)
    docs = vector_store.similarity_search(query_str, k=1)
    return docs[0].page_content
