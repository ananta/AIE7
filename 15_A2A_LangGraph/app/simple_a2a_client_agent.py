#!/usr/bin/env python
"""
Simple A2A Client Agent

A simple client agent that can make API calls to the main A2A Agent Node through the A2A protocol.
This agent uses LangGraph to create a ReAct agent that delegates the heavy lifting to your A2A server.
"""

import os
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import httpx
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import the proper A2A client
from a2a.client import A2AClient as A2AProtocolClient
from a2a.types import MessageSendParams, SendMessageRequest

load_dotenv()


# ---------------------------------------------------------------------
# A2A client (minimal) – adjust endpoints if your routes differ.
# Works with A2AStarletteApplication(DefaultRequestHandler) defaults.
# ---------------------------------------------------------------------

class A2AMessage(BaseModel):
    """User message payload understood by your A2A server."""
    role: str = "user"
    content_type: str = "text/plain"
    content: str


class SimpleA2AClient:
    """Simple wrapper around the proper A2A client."""
    
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.httpx_client = httpx.AsyncClient(timeout=timeout)
        self.a2a_client = None
        
    async def _get_a2a_client(self):
        """Get or create the A2A client."""
        if self.a2a_client is None:
            # We need to get the agent card first
            from a2a.client import A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url,
            )
            agent_card = await resolver.get_agent_card()
            self.a2a_client = A2AProtocolClient(
                httpx_client=self.httpx_client,
                agent_card=agent_card
            )
        return self.a2a_client
        
    async def send_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Send a message using the proper A2A protocol."""
        a2a_client = await self._get_a2a_client()
        
        # Create the proper A2A message format
        send_params = MessageSendParams(
            message={
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message.content}
                ],
                'message_id': str(uuid.uuid4()),
            }
        )
        
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=send_params
        )
        
        # Send the message
        response = await a2a_client.send_message(request)
        
        # Extract the response
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            result = response.root.result
            return {
                'id': getattr(result, 'id', None),
                'context_id': getattr(result, 'context_id', None),
                'state': getattr(result, 'status', None),
                'messages': getattr(result, 'messages', []),
                'artifacts': getattr(result, 'artifacts', [])
            }
        else:
            return {'error': 'Unexpected response format'}
    
    async def aclose(self):
        """Close the client."""
        if self.httpx_client:
            await self.httpx_client.aclose()


# ---------------------------------------------------------------------
# Tool: wraps an A2A call so LangGraph agent can "use" your application
# ---------------------------------------------------------------------

class A2AQueryInput(BaseModel):
    query: str = Field(..., description="The user query to send to the A2A Agent Node.")
    context_id: Optional[str] = Field(
        default=None,
        description="Optional thread/context id to preserve memory."
    )


@tool("a2a_query", args_schema=A2AQueryInput, return_direct=True)
async def a2a_query_tool(query: str, context_id: Optional[str] = None) -> str:
    """
    Call the running A2A Agent Node and return the final answer.
    Streams progress logs (if events are exposed) and waits until completion.
    """
    print(f"[A2A Tool] Processing query: {query}")
    
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:10000")
    client = SimpleA2AClient(base_url)

    try:
        # Create message (A2A expects a user message)
        message = A2AMessage(content=query)

        print(f"[A2A] Sending message to server...")
        
        # Send message using proper A2A protocol
        result = await client.send_message(message)

        # Extract response from the result
        if isinstance(result, dict):
            # Check for errors
            if 'error' in result:
                error_msg = result['error']
                print(f"[A2A Error] {error_msg}")
                return f"Error: {error_msg}"
            
            # Try to extract from messages
            messages = result.get('messages', [])
            if messages:
                for msg in reversed(messages):
                    if isinstance(msg, dict):
                        # Look for content in various possible formats
                        for field in ['content', 'text', 'message']:
                            if field in msg:
                                content = msg[field]
                                if content:
                                    return str(content)
            
            # Try artifacts
            artifacts = result.get('artifacts', [])
            if artifacts:
                for artifact in reversed(artifacts):
                    if isinstance(artifact, dict):
                        parts = artifact.get('parts', [])
                        if parts:
                            for part in parts:
                                if isinstance(part, dict):
                                    for field in ['text', 'content']:
                                        if field in part:
                                            return str(part[field])

        # Last fallback: return the result as string
        print(f"[DEBUG] Full result structure: {result}")
        return f"[A2A Response: {result}]"

    finally:
        await client.aclose()


# ---------------------------------------------------------------------
# Build a tiny ReAct agent that prefers using the A2A tool
# ---------------------------------------------------------------------

def build_simple_agent():
    # Simple agent that directly uses the A2A tool
    # We don't need the ReAct agent for this simple use case
    return a2a_query_tool


# ---------------------------------------------------------------------
# Run from CLI
# ---------------------------------------------------------------------

async def _amain():
    agent = build_simple_agent()
    # Example question – the tool will call your A2A server and stream progress to stdout
    question = "Summarize the latest LLM safety news in 3 bullets."
    
    print(f"Question: {question}")
    print("Processing...")
    
    # Call the tool directly
    result = await agent.ainvoke({"query": question})
    
    print(f"Answer: {result}")


if __name__ == "__main__":
    asyncio.run(_amain())
