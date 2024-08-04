from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from project.use_cases.query_confluence import QueryConfluenceUseCase
from project.use_cases.perform_web_search_with_llm import PerformWebSearchUseCase

GPT_4o_MINI_BASE = ChatOpenAI(model="gpt-4o-mini")

@tool
def query_confluence(query: str):
    """Searches the company's internal knowledge base
        to find the answer to the user's question. The company
        is a software product development company called Vinta Software. 
        Use this tool for anything that looks like a company internal question.
        Args:
            query: user's question
           
    """
    result =  QueryConfluenceUseCase.run_as_tool(query, GPT_4o_MINI_BASE)
    return result.get("answer") # TODO: format a decent answer with the links

@tool
def web_search(query: str):
    """Searches the web to find the answer to the user's question. Used only
       if the query is not related to the company's internal knowledge base.
       The company is a software product development company called Vinta. You 
       should use this tool sparingly. In doubt, answer the question asking the
       user for follow-up information. 

        Args:
            query: user's question
           
    """
    return PerformWebSearchUseCase.run_as_tool(query)

tools = [query_confluence, web_search]

GPT_4o_MINI = GPT_4o_MINI_BASE.bind_tools(tools)