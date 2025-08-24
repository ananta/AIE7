# langgraph_agent_lib/guarded_agents.py
"""
Production-safe LangGraph agent with pre/post guardrails.

Usage:
    from langgraph_agent_lib.guarded_agents import create_guarded_agent
    guarded = create_guarded_agent(
        model_name="gpt-4o-mini",
        temperature=0.2,
        rag_chain=rag_chain,  # optional, improves factuality guard
        allowed_topics=["student loans", "direct loan program", "federal aid"],
        judge_model_name="gpt-4.1-mini",
        max_refine_loops=1,
    )
    out = guarded.invoke({"messages": [HumanMessage(content="What is the main purpose of the Direct Loan Program?")]})
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List, Tuple

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .models import get_openai_model
from .rag import ProductionRAGChain  # for type hints / optional use
from .agents import get_default_tools, AgentState  # reuse your toolbelt + state


__all__ = ["create_guarded_agent"]


# -----------------------------
# Lightweight heuristic guards
# -----------------------------
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # credit card-like
]

_JAILBREAK_HINTS = [
    "ignore previous", "system prompt", "developer mode",
    "break your rules", "bypass", "jailbreak", "as dan", "do anything now"
]


def _pii_hits(text: str) -> List[str]:
    hits = []
    for pat in _PII_PATTERNS:
        hits.extend(pat.findall(text or ""))
    # redact preview
    return [h[:8] + "…" for h in hits]


def _looks_like_jailbreak(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _JAILBREAK_HINTS)


def _on_topic_simple(text: str, allowed_topics: Optional[List[str]]) -> bool:
    if not allowed_topics:
        return True
    t = (text or "").lower()
    return any(k.lower() in t for k in allowed_topics)


# -----------------------------
# LLM-based guard judges
# -----------------------------
def _build_guard_judges(help_model_name: str):
    judge = get_openai_model(model_name=help_model_name, temperature=0.0)
    to_text = StrOutputParser()

    topic_prompt = PromptTemplate.from_template(
        "Allowed topics: {topics}\n\nUser input: {user}\n\n"
        "Answer with a single char: 'Y' if the input is on-topic, else 'N'."
    )
    moderation_prompt = PromptTemplate.from_template(
        "Classify whether the assistant response is SAFE or UNSAFE considering hate, sexual minors, self-harm, "
        "illicit instructions, and malware creation. Reply with 'SAFE' or 'UNSAFE' only.\n\nResponse:\n{resp}"
    )
    factuality_prompt = PromptTemplate.from_template(
        "Given a user question, assistant answer, and reference context excerpts, reply 'Y' if the answer is "
        "well-supported by the context (or admits uncertainty). Otherwise reply 'N'.\n\n"
        "Question: {q}\nAnswer: {a}\nContext:\n{ctx}"
    )

    return {
        "topic": topic_prompt | judge | to_text,
        "moderation": moderation_prompt | judge | to_text,
        "factuality": factuality_prompt | judge | to_text,
    }


def _gather_context_for_factuality(rag_chain: Optional[ProductionRAGChain], question: str, k: int = 3) -> str:
    """Pull a few chunks to let the factuality judge compare answer vs. context."""
    if not rag_chain:
        return ""
    try:
        docs = rag_chain.get_retriever().get_relevant_documents(question)[:k]
    except Exception:
        return ""
    chunks = []
    for d in docs:
        src = d.metadata.get("source", "")
        chunks.append(f"[{src}] {d.page_content[:500]}")
    return "\n---\n".join(chunks)


def _strip_internal(msgs: List[BaseMessage], tags: Tuple[str, ...]) -> List[BaseMessage]:
    """Remove internal control messages from the LLM context."""
    def ok(m: BaseMessage) -> bool:
        c = str(getattr(m, "content", ""))
        return not any(c.startswith(tag) for tag in tags)
    return [m for m in msgs if ok(m)]


# -----------------------------
# Guarded agent factory
# -----------------------------
def create_guarded_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    allowed_topics: Optional[List[str]] = None,   # e.g., ["student loans", "direct loan program"]
    judge_model_name: str = "gpt-4.1-mini",
    max_refine_loops: int = 1,
):
    """
    Create an agent with:
      - PRE-GUARD: jailbreak/PII/off-topic checks (heuristics + optional topic judge)
      - AGENT + TOOLS
      - POST-GUARD: moderation + factuality judge; one refinement loop if needed

    Control markers (internal):
      GUARD:BLOCK: <reason>
      GUARD:REFINE: <hint>
      GUARD:OK
      GUARD:END
    """
    if tools is None:
        tools = get_default_tools(rag_chain)

    agent_model = get_openai_model(model_name=model_name, temperature=temperature).bind_tools(tools)
    judges = _build_guard_judges(judge_model_name)

    # ---------- Nodes ----------
    def pre_guard(state: AgentState) -> Dict[str, Any]:
        user_msg = next((m for m in state["messages"] if isinstance(m, HumanMessage)), state["messages"][0])
        user = str(getattr(user_msg, "content", ""))

        problems = []
        if _looks_like_jailbreak(user):
            problems.append("jailbreak-like instructions detected")
        pii = _pii_hits(user)
        if pii:
            problems.append(f"possible PII in input: {', '.join(pii)}")

        topic_ok = _on_topic_simple(user, allowed_topics)
        if allowed_topics:
            try:
                dec = (judges["topic"].invoke({"topics": ", ".join(allowed_topics), "user": user}) or "").strip().upper()
                topic_ok = dec.startswith("Y")
            except Exception:
                # fall back to heuristic result
                pass
        if not topic_ok:
            problems.append("off-topic query")

        if problems:
            msg = ("GUARD:BLOCK: " + "; ".join(problems) +
                   ". Please rephrase within allowed topics and without sensitive data.")
            return {"messages": [AIMessage(content=msg)]}
        return {}  # pass-through

    def call_model(state: AgentState) -> Dict[str, Any]:
        msgs = state["messages"]
        # Convert last GUARD:REFINE hint (if any) to a SystemMessage
        sys_hints = [m for m in msgs if isinstance(m, AIMessage) and str(m.content).startswith("GUARD:REFINE:")]
        system_prefix = []
        if sys_hints:
            hint = sys_hints[-1].content.replace("GUARD:REFINE:", "").strip()
            system_prefix = [SystemMessage(content=f"Revise per guard feedback: {hint}")]
        cleaned = _strip_internal(msgs, ("GUARD:", "HELPFULNESS:"))
        response = agent_model.invoke(system_prefix + cleaned)
        return {"messages": [response]}

    def route_action_or_postguard(state: AgentState):
        last = state["messages"][-1]
        return "action" if getattr(last, "tool_calls", None) else "post_guard"

    def post_guard(state: AgentState) -> Dict[str, Any]:
        last_ai = next((m for m in reversed(state["messages"])
                        if isinstance(m, AIMessage) and not str(m.content).startswith("GUARD:")), None)
        user_msg = next((m for m in state["messages"] if isinstance(m, HumanMessage)), None)
        if not last_ai or not user_msg:
            return {"messages": [AIMessage(content="GUARD:END")]}  # nothing to judge

        ans = str(last_ai.content)
        q = str(user_msg.content)

        # Moderation
        try:
            mod = (judges["moderation"].invoke({"resp": ans}) or "").strip().upper()
        except Exception:
            mod = "SAFE"

        # Factuality (RAG-enabled)
        try:
            ctx = _gather_context_for_factuality(rag_chain, q, k=3)
            if ctx:
                fact = (judges["factuality"].invoke({"q": q, "a": ans, "ctx": ctx}) or "").strip().upper()
            else:
                fact = "Y"
        except Exception:
            fact = "Y"

        if mod != "SAFE":
            return {"messages": [AIMessage(content="GUARD:BLOCK: unsafe content detected. Please rewrite safely or decline.")]}
        if fact != "Y" and rag_chain is not None:
            return {"messages": [AIMessage(content="GUARD:REFINE: ensure all claims are grounded in the provided document context; if unknown, say 'I don't know'. Include pointers to source_* chunks when possible.")]}
        return {"messages": [AIMessage(content="GUARD:OK")]}

    def post_guard_decision(state: AgentState):
        last = state["messages"][-1]
        text = str(getattr(last, "content", ""))
        if text.startswith("GUARD:BLOCK:") or text == "GUARD:OK":
            return "end"
        # limit refinements
        loops = sum(1 for m in state["messages"]
                    if isinstance(m, AIMessage) and str(getattr(m, "content", "")).startswith("GUARD:REFINE:"))
        if loops >= max_refine_loops:
            return "end"
        return "refine"

    def refine(_state: AgentState) -> Dict[str, Any]:
        # No-op: the presence of GUARD:REFINE is enough; call_model will use it as a SystemMessage.
        return {}

    # ---------- Graph ----------
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    graph.add_node("pre_guard", pre_guard)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("post_guard", post_guard)
    graph.add_node("refine", refine)

    graph.set_entry_point("pre_guard")

    # If pre-guard blocked, END; else continue to agent
    def pre_guard_route(state: AgentState):
        last = state["messages"][-1]
        return END if (isinstance(last, AIMessage) and str(last.content).startswith("GUARD:BLOCK:")) else "agent"

    graph.add_conditional_edges("pre_guard", pre_guard_route, {"agent": "agent", END: END})
    graph.add_conditional_edges("agent", route_action_or_postguard, {"action": "action", "post_guard": "post_guard"})
    graph.add_edge("action", "agent")
    graph.add_conditional_edges("post_guard", post_guard_decision, {"refine": "agent", "end": END})

    return graph.compile()
