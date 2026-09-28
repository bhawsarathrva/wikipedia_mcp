"""Streamlit Chat UI for the Wikipedia MCP Agent.

Features:
- Conversational chat interface using st.chat_message and st.chat_input.
- Persistent session memory via thread_id passed to FastAPI backend.
- Visual display of Model Context Protocol (MCP) tool execution:
    - 🔍 Searching Wikipedia... (tool name and query)
    - ✓ Wikipedia results retrieved
    - 🤖 Final Answer with source citations
- Sidebar with system health diagnostics and one-click sample queries.
"""

import os
import sys
import uuid
from typing import Any, Dict, List
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Wikipedia MCP Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API endpoint configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_backend_health() -> Dict[str, Any]:
    """Queries the FastAPI /health endpoint."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "unreachable", "mcp_server_connected": False, "llm_provider": "unknown"}


def send_chat_message(message: str, thread_id: str) -> Dict[str, Any]:
    """Sends a user query to the FastAPI /chat endpoint."""
    try:
        payload = {"message": message, "thread_id": thread_id}
        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        return {
            "answer": f"Backend returned error {resp.status_code}: {resp.text}",
            "sources": [],
            "tool_used": False,
            "tool_calls": [],
            "thread_id": thread_id,
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": (
                "⚠️ **Could not connect to FastAPI backend.**\n\n"
                "Please make sure the backend server is running:\n"
                "```bash\nuvicorn app.main:app --reload --port 8000\n```"
            ),
            "sources": [],
            "tool_used": False,
            "tool_calls": [],
            "thread_id": thread_id,
        }
    except Exception as exc:
        return {
            "answer": f"⚠️ Unexpected client error: {str(exc)}",
            "sources": [],
            "tool_used": False,
            "tool_calls": [],
            "thread_id": thread_id,
        }


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"

# Sidebar UI
with st.sidebar:
    st.title("🌐 System Status")

    health = check_backend_health()
    if health["status"] == "healthy":
        st.success("🟢 FastAPI Backend: Online")
        if health.get("mcp_server_connected"):
            st.success("🟢 Wikipedia MCP Server: Connected")
        else:
            st.warning("🟡 MCP Server: Connecting...")
        st.info(f"🧠 LLM Provider: **{health.get('llm_provider', 'gemini').upper()}**")
    else:
        st.error("🔴 FastAPI Backend: Offline")
        st.caption(f"Target URL: `{BACKEND_URL}`")

    st.divider()

    st.subheader("💬 Conversation Session")
    st.caption(f"Session Thread ID: `{st.session_state.thread_id}`")
    if st.button("🔄 Reset Conversation / New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.divider()

    st.subheader("💡 Sample Questions")
    samples = [
        "Who was Alan Turing?",
        "When was he born?",
        "What is quantum mechanics?",
        "When was the Eiffel Tower completed?",
        "Hello! How are you?",
    ]
    for sample in samples:
        if st.button(sample, key=f"sample_{sample}", use_container_width=True):
            st.session_state.sample_query = sample
            st.rerun()

    st.divider()
    st.caption("Powered by **LangGraph + Model Context Protocol (MCP)**")


# Main Chat Interface Header
st.title("🌐 Wikipedia MCP Research Assistant")
st.markdown(
    "Ask factual questions grounded in live Wikipedia content via official "
    "**Model Context Protocol (MCP)** tools and **LangGraph** orchestration."
)

# Render Chat History
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        # If assistant used MCP tools, render tool call visualization box
        if role == "assistant" and msg.get("tool_calls"):
            with st.expander("🔍 Model Context Protocol (MCP) Tool Execution", expanded=False):
                for tc in msg["tool_calls"]:
                    tool_name = tc.get("tool", "search_wikipedia")
                    query = tc.get("query", "")
                    snippet = tc.get("result_snippet", "")
                    st.markdown(f"**Tool Invoked:** `{tool_name}`")
                    if query:
                        st.markdown(f"**Query Parameter:** *\"{query}\"*")
                    if tc.get("arguments"):
                        st.code(str(tc["arguments"]), language="json")
                    st.markdown("✅ *Wikipedia results retrieved through MCP Server*")
                    if snippet:
                        st.caption(f"Preview: {snippet}")

        st.markdown(msg["content"])

        # Display Sources if available
        if role == "assistant" and msg.get("sources"):
            st.markdown("##### 📚 Sources")
            for src in msg["sources"]:
                st.markdown(f"- [{src}]({src})")


# Check if user clicked a sample question from sidebar
user_prompt = None
if "sample_query" in st.session_state and st.session_state.sample_query:
    user_prompt = st.session_state.sample_query
    del st.session_state.sample_query

# Input Bar
chat_input = st.chat_input("Ask a question about history, science, people, or concepts...")
if chat_input:
    user_prompt = chat_input

if user_prompt:
    # 1. Append and display User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Invoke Assistant
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Agent reasoning and consulting Wikipedia MCP tools...*")

        response = send_chat_message(user_prompt, thread_id=st.session_state.thread_id)
        status_placeholder.empty()

        answer = response.get("answer", "No response generated.")
        sources = response.get("sources", [])
        tool_used = response.get("tool_used", False)
        tool_calls = response.get("tool_calls", [])

        # Display MCP tool activity if any tools were used
        if tool_used and tool_calls:
            with st.expander("🔍 Model Context Protocol (MCP) Tool Execution", expanded=True):
                for tc in tool_calls:
                    tool_name = tc.get("tool", "search_wikipedia")
                    query = tc.get("query", "")
                    st.markdown(f"**🔍 Tool:** `{tool_name}`")
                    if query:
                        st.markdown(f"**Query:** *\"{query}\"*")
                    st.markdown("✓ **Wikipedia results retrieved via MCP STDIO transport**")
                    if tc.get("result_snippet"):
                        st.caption(f"Data snippet: {tc['result_snippet']}")

        st.markdown(answer)

        if sources:
            st.markdown("##### 📚 Sources")
            for src in sources:
                st.markdown(f"- [{src}]({src})")

        # Save to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tool_calls": tool_calls,
            "sources": sources,
        })
