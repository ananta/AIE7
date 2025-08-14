# app.py
import asyncio
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()

# --- Config ---
SERVER_PATH = Path(__file__).parent / "server.py"

# Choose a model. Requires `langchain-openai` (or swap to anthropic).
# export OPENAI_API_KEY=...
MODEL_ID = "openai:gpt-4o-mini"  # or "anthropic:claude-3-7-sonnet-latest"

async def build_graph():
    client = MultiServerMCPClient({
        "mcp-server": {
            "command": "python",
            "args": [str(SERVER_PATH)],
            "transport": "stdio",
            "env": {
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                "OPENWEATHER_API_KEY": os.getenv("OPENWEATHER_API_KEY", ""),
            }
        }
    })

    tools = await client.get_tools()

    model = init_chat_model(MODEL_ID, temperature=0)
    model_with_tools = model.bind_tools(tools)

    # 4) Tool executor node
    tool_node = ToolNode(tools)

    # 5) Simple controller to decide when to call tools
    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    # 6) Model call node
    async def call_model(state: MessagesState):
        msgs = state["messages"]
        resp = await model_with_tools.ainvoke(msgs)
        return {"messages": [resp]}

    # 7) Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")

    return builder.compile()

async def demo():
    graph = await build_graph()

    tests = [
        "Ping with message 'hello from langgraph'.",
        "Roll dice: 2d6, 3 times.",
        "Give me a fun fact about space.",
        "What's the weather in Austin in imperial units?",
        "Search the web for 'LangGraph MCP adapter quickstart' and summarize briefly."
    ]

    for q in tests:
        print(f"\nUSER: {q}")
        out = await graph.ainvoke({"messages": [{"role": "user", "content": q}]})
        # The last assistant message is at the end
        last = out["messages"][-1]
        print(f"ASSISTANT: {getattr(last, 'content', last)}")

if __name__ == "__main__":
    asyncio.run(demo())
