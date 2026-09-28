from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    tool_results: List[Dict[str, Any]]
    final_answer: str
    sources: List[str]
    tool_used: bool
    tool_calls_info: List[Dict[str, Any]]
