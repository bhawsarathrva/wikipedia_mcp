"""Node functions for the LangGraph Wikipedia MCP Agent workflow."""

import json
import logging
import re
from typing import Any, Callable, Dict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.prompts import load_system_prompt
from app.graph.state import AgentState
logger = logging.getLogger("app.graph.nodes")


def extract_wikipedia_urls(text: str) -> List[str]:
    """Finds Wikipedia URLs in text or JSON strings."""
    pattern = r"https?://[a-zA-Z0-9.\-_/]*wikipedia\.org/wiki/[^\s\",'\]]+"
    matches = re.findall(pattern, text)
    seen = set()
    unique_urls = []
    for url in matches:
        clean_url = url.rstrip(".)")
        if clean_url not in seen:
            seen.add(clean_url)
            unique_urls.append(clean_url)
    return unique_urls


def create_receive_question_node():
    async def receive_question(state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "").strip()
        logger.info(f"[USER] {question}")

        messages: List[BaseMessage] = list(state.get("messages", []))

        has_system = any(isinstance(m, SystemMessage) for m in messages)
        new_messages: List[BaseMessage] = []

        if not has_system:
            sys_prompt = load_system_prompt()
            new_messages.append(SystemMessage(content=sys_prompt))

        if not messages or not (
            isinstance(messages[-1], HumanMessage) and messages[-1].content == question
        ):
            new_messages.append(HumanMessage(content=question))

        return {
            "messages": new_messages,
            "tool_used": state.get("tool_used", False),
            "tool_results": state.get("tool_results", []),
            "tool_calls_info": state.get("tool_calls_info", []),
            "sources": state.get("sources", []),
        }

    return receive_question


def create_call_llm_node(llm_with_tools: Any):
    async def call_llm(state: AgentState) -> Dict[str, Any]:
        logger.info("[LANGGRAPH] Calling LLM")
        messages = state["messages"]

        response: AIMessage = await llm_with_tools.ainvoke(messages)

        if response.tool_calls:
            for tc in response.tool_calls:
                logger.info(f"[LLM] Tool requested: {tc.get('name')} (args={tc.get('args')})")
        else:
            logger.info("[LLM] No tools requested; direct text response generated.")

        return {
            "messages": [response],
        }
    return call_llm


def create_mcp_tool_node(tools: List[BaseTool]):
    tools_by_name: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def mcp_tool_node(state: AgentState) -> Dict[str, Any]:
        """[NODE]: mcp_tool.

        Executes the tool call(s) requested by the LLM by forwarding them
        to the MCP Client, which routes them to the Wikipedia MCP Server.
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.warning("[LANGGRAPH] mcp_tool node invoked without tool_calls in last message.")
            return {}

        tool_messages: List[ToolMessage] = []
        new_tool_results: List[Dict[str, Any]] = list(state.get("tool_results", []))
        new_tool_calls_info: List[Dict[str, Any]] = list(state.get("tool_calls_info", []))
        collected_sources: List[str] = list(state.get("sources", []))

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", f"call_{tool_name}")

            logger.info(f"[MCP CLIENT] Calling {tool_name}")

            tool_instance = tools_by_name.get(tool_name)
            if not tool_instance:
                err_text = f"Tool '{tool_name}' not available on Wikipedia MCP Server."
                logger.error(f"[LANGGRAPH] {err_text}")
                tool_messages.append(
                    ToolMessage(content=json.dumps({"error": err_text}), tool_call_id=tool_call_id)
                )
                continue

            try:
                result_output = await tool_instance.ainvoke(tool_args)
                result_str = str(result_output)
                logger.info(f"[MCP CLIENT] Tool result received for {tool_name}")

                found_urls = extract_wikipedia_urls(result_str)
                for u in found_urls:
                    if u not in collected_sources:
                        collected_sources.append(u)

                query_extracted = tool_args.get("query") or tool_args.get("title")
                snippet = result_str[:150].replace("\n", " ") + "..."

                new_tool_calls_info.append({
                    "tool": tool_name,
                    "query": query_extracted,
                    "arguments": tool_args,
                    "result_snippet": snippet,
                })

                new_tool_results.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "output": result_str,
                })

                tool_messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_call_id)
                )

            except Exception as exc:
                logger.error(f"[MCP CLIENT] Error executing {tool_name}: {exc}", exc_info=True)
                err_msg = json.dumps({"error": f"Failed to execute MCP tool: {str(exc)}"})
                tool_messages.append(
                    ToolMessage(content=err_msg, tool_call_id=tool_call_id)
                )

        return {
            "messages": tool_messages,
            "tool_used": True,
            "tool_results": new_tool_results,
            "tool_calls_info": new_tool_calls_info,
            "sources": collected_sources,
        }

    return mcp_tool_node


def create_final_answer_node():
    """Factory creating the final_answer formatting node."""

    async def final_answer(state: AgentState) -> Dict[str, Any]:
        """[NODE]: final_answer.

        Extracts the final response from the last AIMessage, gathers canonical
        sources, and prepares the final state.
        """
        logger.info("[LLM] Generating final answer")
        last_message = state["messages"][-1]
        answer_text = ""

        if isinstance(last_message, AIMessage):
            if isinstance(last_message.content, list):
                parts = []
                for part in last_message.content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                answer_text = "\n".join(parts)
            else:
                answer_text = str(last_message.content)
        else:
            answer_text = str(last_message.content)

        text_urls = extract_wikipedia_urls(answer_text)
        sources: List[str] = list(state.get("sources", []))
        for u in text_urls:
            if u not in sources:
                sources.append(u)

        return {
            "final_answer": answer_text,
            "sources": sources,
        }

    return final_answer
