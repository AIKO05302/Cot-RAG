import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

DATA_PATH = r".\data\week1"
embeddings = OllamaEmbeddings(base_url="http://127.0.0.1:11434",model="nomic-embed-text")
vector_store = Chroma(
            collection_name="foo",
            embedding_function=embeddings,
            persist_directory="./chroma_db"  # 保存路径
        )
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


#读取每一个txt文件内容
def embedded_text(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1200,  # 每个块的大小
        chunk_overlap=100  # 块之间的重叠部分
    )
    chunks = text_splitter.split_text(text)
    # 查看切块结果
    for i,chunk in enumerate(chunks):
        print(f"{i+1}/{len(chunks)}")
        document = Document(page_content=chunk)
        vector_store.add_documents([document])



def init_index(WEEK):
    with open(WEEK, "r", encoding="utf-8") as f:
        text = f.read()
        embedded_text(text)
WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    #加载原始文件并保存到chroma
    init_index(WEEK)
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")