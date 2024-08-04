from langchain_core.messages import HumanMessage, ToolMessage
from project.llm.gpt import query_confluence, web_search


class AnswerSlackMessageUseCase:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools

    def execute(self, query):
        messages = [HumanMessage(query)]
        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            selected_tool = {
                "query_confluence": query_confluence,
                "web_search": web_search,
            }[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        return messages
