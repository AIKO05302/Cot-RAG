import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from Tools import mocked_location_tool, calculator_tool, calendar_tool, webpage_tool, file_toolkit
from AutoCot.CotLLM import CotLLM

load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

tools = [
    mocked_location_tool,
    calculator_tool,
    calendar_tool,
    webpage_tool,
] + file_toolkit.get_tools()


def launch_agent(agent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    while True:
        task = input(f"{ai_icon}有什么可以帮您:\n{human_icon}:>>>")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task,verbose=True)
        print(f"{ai_icon}: {reply}\n")


def main():
    llm = ChatOllama(model="qwen2.5:14b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="my_collection",
    )
    retriever = vector_store.as_retriever(search_kwargs=dict(k=1))
    agent = CotLLM(
        llm=llm,
        prompts_path="./prompts/main",
        tools=tools,
        work_dir = ".",
        main_prompt_file = "main.json",
        final_prompt_file = "final_step.json",
        max_thought_steps=20, # 最大推理步数
        memery_retriever=retriever
    )
    launch_agent(agent)


if __name__ == "__main__":
    main()
