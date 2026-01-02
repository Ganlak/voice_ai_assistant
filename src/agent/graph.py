"""
Agent Graph - MVP Production Grade (Pure LangChain/LangGraph)
=============================================================
LangGraph agent for Voice AI using SOP documents.

Features:
- Pure LangChain/LangGraph implementation
- Tool-based SOP retrieval
- Conversation state management
- Voice-optimized responses

Usage:
    from src.agent.graph import run_agent, agent
    
    response = run_agent("How do I schedule an appointment?")

Author: ThubIQ Healthcare
Version: 2.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Literal

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Local imports
from src.agent.state import (
    AgentState,
    create_initial_state,
    create_state_from_query,
    update_context,
    update_response,
    update_error,
    update_call_stage,
    add_tool_call,
    has_error,
    has_context,
    get_last_user_message
)
from src.agent.prompts import SYSTEM_PROMPT, PROMPTS, get_prompt
from src.tools.search_sop import search_sop_tool, get_procedure, sop_tools


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

def get_llm() -> AzureChatOpenAI:
    """Get Azure OpenAI LLM instance."""
    
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        temperature=0.7,
        max_tokens=300  # Keep responses short for voice
    )


# =============================================================================
# LLM WITH TOOLS
# =============================================================================

def get_llm_with_tools():
    """Get LLM bound with SOP tools."""
    
    llm = get_llm()
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(sop_tools)
    
    return llm_with_tools


# =============================================================================
# GRAPH NODES
# =============================================================================

def agent_node(state: AgentState) -> AgentState:
    """
    Main agent node - processes user input and decides action.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with AI response or tool calls
    """
    logger.info("Agent node: Processing user input")
    
    try:
        # Get LLM with tools
        llm = get_llm_with_tools()
        
        # Build messages
        messages = list(state["messages"])
        
        # Add system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        
        # Invoke LLM
        response = llm.invoke(messages)
        
        # Update state with response
        new_messages = list(state["messages"]) + [response]
        
        return {
            **state,
            "messages": new_messages,
            "response": response.content if hasattr(response, 'content') else str(response)
        }
        
    except Exception as e:
        logger.error(f"Agent node error: {str(e)}")
        return {
            **state,
            "error": str(e),
            "response": get_prompt("error")
        }


def tool_node(state: AgentState) -> AgentState:
    """
    Tool execution node - runs tools called by agent.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool results
    """
    logger.info("Tool node: Executing tools")
    
    try:
        # Get the last message (should have tool calls)
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("No tool calls found")
            return state
        
        # Create tool node with our tools
        tool_executor = ToolNode(sop_tools)
        
        # Execute tools
        result = tool_executor.invoke(state)
        
        # Merge results into state
        if "messages" in result:
            new_messages = list(state["messages"]) + list(result["messages"])
            return {
                **state,
                "messages": new_messages,
                "context": str(result["messages"][-1].content) if result["messages"] else None
            }
        
        return state
        
    except Exception as e:
        logger.error(f"Tool node error: {str(e)}")
        return {
            **state,
            "error": str(e)
        }


def response_node(state: AgentState) -> AgentState:
    """
    Response generation node - creates final voice response.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final response
    """
    logger.info("Response node: Generating final response")
    
    try:
        # Get the last AI message
        last_message = state["messages"][-1]
        
        # Extract content
        if hasattr(last_message, 'content') and last_message.content:
            response = last_message.content
        else:
            response = get_prompt("error")
        
        return {
            **state,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Response node error: {str(e)}")
        return {
            **state,
            "error": str(e),
            "response": get_prompt("error")
        }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_use_tools(state: AgentState) -> Literal["tools", "response"]:
    """
    Determine if tools should be used.
    
    Args:
        state: Current agent state
        
    Returns:
        "tools" if tool calls exist, "response" otherwise
    """
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Routing to tools")
        return "tools"
    
    logger.info("Routing to response")
    return "response"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph agent.
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Building agent graph")
    
    # Create graph with state schema
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("response", response_node)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            "response": "response"
        }
    )
    
    # Add edge from tools back to agent
    graph.add_edge("tools", "agent")
    
    # Add edge from response to END
    graph.add_edge("response", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("Agent graph built successfully")
    
    return compiled_graph


# =============================================================================
# AGENT INSTANCE
# =============================================================================

# Build agent graph (singleton)
agent = build_agent_graph()


# =============================================================================
# AGENT EXECUTION FUNCTIONS
# =============================================================================

def run_agent(query: str, system_prompt: str = None) -> str:
    """
    Run the agent with a user query.
    
    Args:
        query: User's query
        system_prompt: Optional custom system prompt
        
    Returns:
        Agent's response string
    """
    logger.info(f"Running agent with query: {query[:50]}...")
    
    try:
        # Create initial state
        messages = []
        
        # Add system prompt
        prompt = system_prompt or SYSTEM_PROMPT
        messages.append(SystemMessage(content=prompt))
        
        # Add user message
        messages.append(HumanMessage(content=query))
        
        # Create state
        initial_state = AgentState(
            messages=messages,
            context=None,
            current_query=query,
            tool_calls=[],
            response=None,
            error=None,
            call_stage="main"
        )
        
        # Run graph
        result = agent.invoke(initial_state)
        
        # Extract response
        response = result.get("response", get_prompt("error"))
        
        logger.info(f"Agent response: {response[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        return get_prompt("error")


def run_agent_with_history(
    query: str,
    history: list = None,
    system_prompt: str = None
) -> tuple:
    """
    Run the agent with conversation history.
    
    Args:
        query: User's current query
        history: List of previous messages
        system_prompt: Optional custom system prompt
        
    Returns:
        Tuple of (response, updated_history)
    """
    logger.info(f"Running agent with history: {len(history or [])} messages")
    
    try:
        # Build messages
        messages = []
        
        # Add system prompt
        prompt = system_prompt or SYSTEM_PROMPT
        messages.append(SystemMessage(content=prompt))
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current query
        messages.append(HumanMessage(content=query))
        
        # Create state
        state = AgentState(
            messages=messages,
            context=None,
            current_query=query,
            tool_calls=[],
            response=None,
            error=None,
            call_stage="main"
        )
        
        # Run graph
        result = agent.invoke(state)
        
        # Extract response
        response = result.get("response", get_prompt("error"))
        
        # Build updated history (exclude system message)
        updated_history = [m for m in result["messages"] if not isinstance(m, SystemMessage)]
        
        return response, updated_history
        
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        return get_prompt("error"), history or []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chat(query: str) -> str:
    """
    Simple chat function for quick testing.
    
    Args:
        query: User's query
        
    Returns:
        Agent's response
    """
    return run_agent(query)


def get_greeting() -> str:
    """
    Get the agent's greeting message.
    
    Returns:
        Greeting string
    """
    return "Thank you for calling WellStreet Urgent Care. This is Sarah. How may I help you today?"


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test agent graph."""
    
    print("=" * 70)
    print("Agent Graph Test (LangGraph)")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        "Hello, I need to schedule an appointment.",
        "Do you accept walk-ins?",
        "What's your policy for late arrivals?",
        "How do I get to your clinic?",
        "What are your wait times like?"
    ]
    
    print("\nRunning test queries:")
    print("-" * 70)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        
        response = run_agent(query)
        
        print(f"Sarah: {response}")
        print("-" * 50)
    
    # Test with history
    print("\n" + "=" * 70)
    print("Testing conversation with history:")
    print("=" * 70)
    
    history = []
    
    conversation = [
        "Hi, I'd like to schedule an appointment.",
        "Can I also just walk in instead?",
        "What if I'm running late?"
    ]
    
    for query in conversation:
        print(f"\nUser: {query}")
        
        response, history = run_agent_with_history(query, history)
        
        print(f"Sarah: {response}")
        print(f"   [History: {len(history)} messages]")
    
    print("\n" + "=" * 70)
    print("Agent Graph Test Complete")
    print("=" * 70)