"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    helpfulness_model_name: str = "gpt-4.1-mini",
    max_helpfulness_loops: int = 3,
):
    """
    Agent that, after each response, runs a lightweight 'helpfulness' evaluator.
    If evaluator returns 'Y', end. If 'N', loop back to the agent (up to a safeguard limit).
    """
    if tools is None:
        tools = get_default_tools(rag_chain)

    # Main model (tool-enabled)
    agent_model = get_openai_model(model_name=model_name, temperature=temperature).bind_tools(tools)

    # Small/cheap judge
    judge_model = get_openai_model(model_name=helpfulness_model_name, temperature=0.0)
    judge_prompt = PromptTemplate.from_template(
        """You will judge if the final response is extremely helpful for the user's initial query.
            Reply with a single character: 'Y' for helpful, 'N' for not helpful.

        Initial Query:
        {initial_query}

        Final Response:
        {final_response}
        """
    )
    judge_chain = judge_prompt | judge_model | StrOutputParser()

    def _filtered_messages_for_agent(msgs: List[AIMessage]):
        """Hide internal 'HELPFULNESS:*' control messages from the LLM."""
        return [m for m in msgs if not (getattr(m, "content", "").startswith("HELPFULNESS:"))]

    def _helpfulness_iterations(msgs: List[AIMessage]) -> int:
        return sum(1 for m in msgs if isinstance(m, AIMessage) and str(getattr(m, "content", "")).startswith("HELPFULNESS:"))

    # --- Nodes ---
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke main model with tools, ignoring internal helpfulness markers."""
        messages = _filtered_messages_for_agent(state["messages"])
        response = agent_model.invoke(messages)
        return {"messages": [response]}

    def route_to_action_or_helpfulness(state: AgentState):
        """If the last message has tool calls, go to tools; else evaluate helpfulness."""
        last = state["messages"][-1]
        return "action" if getattr(last, "tool_calls", None) else "helpfulness"

    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Judge the last response vs the initial query; emit HELPFULNESS:Y/N/END."""
        # Guardrail against infinite loops
        if _helpfulness_iterations(state["messages"]) >= max_helpfulness_loops:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}

        # Initial query (first user content) & latest model reply
        initial_query_text = getattr(state["messages"][0], "content", "")
        final_response_text = getattr(state["messages"][-1], "content", "")

        decision_raw = judge_chain.invoke(
            {"initial_query": initial_query_text, "final_response": final_response_text}
        )
        decision_char = (decision_raw or "").strip().upper()[:1]
        decision = "Y" if decision_char == "Y" else "N"

        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Map HELPFULNESS markers to graph edges."""
        last_text = str(getattr(state["messages"][-1], "content", ""))
        if last_text == "HELPFULNESS:END" or "HELPFULNESS:Y" in last_text:
            return "end"
        return "continue"

    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_edge("action", "agent")
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END},
    )

    return graph.compile()
