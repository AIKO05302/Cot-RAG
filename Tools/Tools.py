import warnings

from pydantic import Field

warnings.filterwarnings("ignore")
from langchain.agents import Tool
from py_expression_eval import Parser
from langchain.tools import StructuredTool
from .WebpageUtil import read_webpage
from langchain.tools import tool
from ctparse import ctparse

@tool("UserLocation")
def mocked_location_tool(foo: str) -> str:
    """用于获取用户当前的位置（城市、区域）"""
    return "Beijing"

@tool("Calendar")
def calendar_tool(
        date_exp: str = Field(description="Date expression to be parsed. It must be in English."),
) -> str:
    """用于查询和计算日期/时间"""
    res = ctparse(date_exp)
    date = res.resolution
    return date.dt.strftime("%c")

def evaluate(expr: str) -> str:
    parser = Parser()
    return str(parser.parse(expr).evaluate({}))


calculator_tool = Tool.from_function(
    func=evaluate,
    name="Calculator",
    description="用于计算一个数学表达式的值",
)


webpage_tool = StructuredTool.from_function(
    func=read_webpage,
    name="ReadWebpage",
    description="用于获取一个网页的内容",
)