import json
import os
import tempfile
from typing import Optional, List
from langchain_core.prompts import BasePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import load_prompt, PipelinePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.tools import BaseTool, Tool

from Utils.ThoughtAndAction import Action,ThoughtAndAction


def _load_file(filename):
    """Loads a file into a string"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    f = open(filename, 'r', encoding='utf-8')
    s = f.read()
    f.close()
    return s


def _chinese_friendly(i_string):
    lines = i_string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)




class PromptTemplateBuilder:
    def __init__(
            self,
            prompt_path: str,
            prompt_file: str = "main.templ",
    ):
        self.prompt_path = prompt_path
        self.prompt_file = prompt_file

    def build(
            self,
            tools: Optional[List[BaseTool]] = None,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        main_file = os.path.join(self.prompt_path, self.prompt_file)
        main_prompt_template = load_prompt(
            self._check_or_redirect(main_file),encoding="utf-8"
        )
        variables = main_prompt_template.input_variables
        partial_variables = {}
        recursive_templates = []
        for var in variables:
            # 是否存在嵌套模板
            if os.path.exists(os.path.join(self.prompt_path, f"{var} json")):
                sub_template = PromptTemplateBuilder(
                    self.prompt_path, f"{var}.json"
                ).build(tools=tools, output_parser=output_parser)
                recursive_templates.append((var, sub_template))
            # 是否存在文本文件
            elif os.path.exists(os.path.join(self.prompt_path, f"{var}.txt")):
                var_str = _load_file(os.path.join(self.prompt_path, f"{var}.txt"))
                partial_variables[var] = var_str

        if tools is not None and "tools" in variables:
            tools_prompt = self._get_tools_prompt(tools)
            partial_variables["tools"] = tools_prompt

        if output_parser is not None and "format_instructions" in variables:
            partial_variables["format_instructions"] = _chinese_friendly(
                output_parser.get_format_instructions()
            )

        if recursive_templates:
            #将有值嵌套的模板填充到主模板
            main_prompt_template = PipelinePromptTemplate(
                final_prompt=main_prompt_template,
                pipeline_prompts=recursive_templates,
            )

        return main_prompt_template.partial(**partial_variables)

    def _check_or_redirect(self, prompt_file):
        with open(prompt_file, 'r', encoding="utf-8") as f:
            config = json.load(f)
        if "template_path" in config:
            # 如果是相对路径，则转换为绝对路径
            if not os.path.isabs(config["template_path"]):
                config["template_path"] = os.path.join(self.prompt_path, config["template_path"])
            # 生成临时文件
            tmp_file = tempfile.NamedTemporaryFile(
                suffix='.json',
                mode="w",
                encoding="utf-8",
                delete=False
            )
            tmp_file.write(json.dumps(config, ensure_ascii=False))
            tmp_file.close()
            return tmp_file.name
        return prompt_file

    def _get_tools_prompt(self, tools):
        tools_prompt = ""
        for i, tool in enumerate(tools):
            prompt = f"{i + 1}. {tool.name}: {tool.description}, \
                            args json schema: {json.dumps(tool.args, ensure_ascii=False)}\n"
            tools_prompt += prompt
        return tools_prompt


if __name__ == "__main__":
    builder = PromptTemplateBuilder("../prompts/main", "main.json")
    output_parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = builder.build(tools=[
        Tool(name="FINISH", func=lambda: None, description="结束任务")
    ], output_parser=output_parser)
    print(prompt_template.format(
        task_description="解决问题",
        short_term_memory="",
        long_term_memory="",
        work_dir=".",
    ))
