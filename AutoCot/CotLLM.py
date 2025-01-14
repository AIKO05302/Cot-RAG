from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models import BaseChatModel
from typing import List, Optional
from langchain.tools.base import BaseTool
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import ValidationError

from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.ThoughtAndAction import Action
import sys
from colorama import Style, Fore


THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.BLUE
CODE_COLOR = Fore.WHITE


def color_print(text, color=None, end='\n'):
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()

def _format_short_term_memory(memory):
    messages = memory.chat_memory.messages
    string_messages = [messages[i].content for i in range(1,len(messages))]
    return "\n".join(string_messages)

def _format_long_term_memory(task_description, memory):
    return memory.load_memory_variables(
        {"prompt": task_description}
    )["history"]


class CotLLM:
    """基于langchain实现"""
    def __init__(
            self,
            llm: BaseChatModel,
            prompts_path: str,
            tools: List[BaseTool],
            work_dir = "./data", # 工作目录
            main_prompt_file = "main.json", # 主要prompt
            final_prompt_file = "final.json", # 最终输出的prompt
            max_thought_steps: Optional[int] = 10, # 最大推理步数
            memery_retriever: Optional[VectorStoreRetriever] = None,
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps
        self.memery_retriever = memery_retriever
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        #自动纠错
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.main_prompt_file = main_prompt_file
        self.final_prompt_file = final_prompt_file

    #主入口
    def run(self, task_description, verbose=False) -> str:
        thought_step_count = 0 # 思考步数
        # 如果有长时记忆，加载长时记忆
        if self.memery_retriever is not None:
            long_term_memory = VectorStoreRetrieverMemory(
                retriever=self.memery_retriever
            )
        else:
            long_term_memory = None
        prompt_template = PromptTemplateBuilder(
            prompt_path=self.prompts_path,
            prompt_file=self.main_prompt_file,
        ).build(
            tools=self.tools,
            output_parser=self.output_parser,
        ).partial(
            work_dir=self.work_dir,
            task_description=task_description,
            long_term_memory=_format_long_term_memory(task_description, long_term_memory)
            if long_term_memory is not None else "",
        )

        short_term_memory = ConversationBufferWindowMemory(
            llm = self.llm,
            max_token_limit = 4000,
        )

        short_term_memory.save_context(
            {"input":"\n初始化"},
            {"output":"\n开始思考"}
        )

        # 初始化链
        chain = prompt_template | self.llm | StrOutputParser()


        reply = ""
        while thought_step_count < self.max_thought_steps:
            if verbose:
                color_print(f">>>Round: {thought_step_count}<<<",ROUND_COLOR)
            action, response = self._step(
                chain,
                short_term_memory=short_term_memory,
                verbose=verbose,
            )
            if action.name == "FINISH":
                if verbose:
                    color_print(f"\n------>>>FINISH<<<", OBSERVATION_COLOR)
                reply = self._final_step(short_term_memory, task_description)
                break
            observation = self._exec_action(action)
            if verbose:
                color_print(f"------>>>结果: \n{observation}<<<", OBSERVATION_COLOR)
            short_term_memory.save_context(
                {"input":response},
                {"output":"返回结果:\n" + observation}
            )
            thought_step_count += 1

            if not reply:
                reply = "抱歉，我没能完成您的任务。"
            if long_term_memory is not None:
                long_term_memory.save_context(
                    {"input": task_description},
                    {"output": reply}
                )
        return reply



    def _step(self, re_chain, short_term_memory, verbose):
        """执行一步思考"""
        response = ""
        for s in re_chain.stream({
            "short_term_memory": _format_short_term_memory(short_term_memory),
        }):
            if verbose:
                color_print(s, THOUGHT_COLOR, end="")
            response += s
        action = self.robust_parser.parse(response)
        return action, response

    def _final_step(self, short_term_memory, task_description):
        finish_prompt = PromptTemplateBuilder(
            self.prompts_path,
            self.final_prompt_file,
        ).build().partial(
            task_description=task_description,
            short_term_memory=_format_short_term_memory(short_term_memory),
        )
        chain = finish_prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        return response

    def _exec_action(self, action):
        # 查找工具
        tool = self._find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = (
                    f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                )
        return observation

    def _find_tool(self, tool_name):
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None










