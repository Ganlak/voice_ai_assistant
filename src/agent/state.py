"""
Agent State - MVP Production Grade (Pure LangChain/LangGraph)
=============================================================
State management for LangGraph agent using TypedDict pattern.

Author: ThubIQ Healthcare
Version: 2.0.0
"""

from typing import Annotated, List, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages


# =============================================================================
# AGENT STATE (LangGraph Pattern)
# =============================================================================

class AgentState(TypedDict):
    """
    LangGraph agent state using TypedDict pattern.
    
    Attributes:
        messages: Conversation history with automatic message merging
        context: Retrieved SOP context from vector search
        current_query: The current user query being processed
        tool_calls: List of tool calls made in current turn
        response: Final response to return to user
        error: Error message if something went wrong
        call_stage: Current stage of the call (greeting, main, closing)
    """
    
    # Message history with LangGraph's add_messages reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Retrieved context from SOP search
    context: Optional[str]
    
    # Current query being processed
    current_query: Optional[str]
    
    # Tool execution tracking
    tool_calls: List[str]
    
    # Response to user
    response: Optional[str]
    
    # Error tracking
    error: Optional[str]
    
    # Call stage tracking
    call_stage: str


# =============================================================================
# STATE FACTORY FUNCTIONS
# =============================================================================

def create_initial_state(
    system_prompt: Optional[str] = None,
    initial_message: Optional[str] = None
) -> AgentState:
    """
    Create initial agent state.
    
    Args:
        system_prompt: Optional system prompt to include
        initial_message: Optional initial user message
        
    Returns:
        Initialized AgentState
    """
    messages: List[BaseMessage] = []
    
    # Add system message if provided
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    # Add initial user message if provided
    if initial_message:
        messages.append(HumanMessage(content=initial_message))
    
    return AgentState(
        messages=messages,
        context=None,
        current_query=initial_message,
        tool_calls=[],
        response=None,
        error=None,
        call_stage="greeting"
    )


def create_state_from_query(
    query: str,
    system_prompt: Optional[str] = None
) -> AgentState:
    """
    Create agent state from a user query.
    
    Args:
        query: User's query
        system_prompt: Optional system prompt
        
    Returns:
        AgentState initialized with query
    """
    return create_initial_state(
        system_prompt=system_prompt,
        initial_message=query
    )


# =============================================================================
# STATE UPDATE FUNCTIONS
# =============================================================================

def update_context(state: AgentState, context: str) -> AgentState:
    """
    Update state with retrieved context.
    
    Args:
        state: Current state
        context: Retrieved SOP context
        
    Returns:
        Updated state
    """
    return AgentState(
        messages=state["messages"],
        context=context,
        current_query=state["current_query"],
        tool_calls=state["tool_calls"],
        response=state["response"],
        error=state["error"],
        call_stage=state["call_stage"]
    )


def update_response(state: AgentState, response: str) -> AgentState:
    """
    Update state with agent response.
    
    Args:
        state: Current state
        response: Agent's response
        
    Returns:
        Updated state with response and AI message
    """
    new_messages = list(state["messages"]) + [AIMessage(content=response)]
    
    return AgentState(
        messages=new_messages,
        context=state["context"],
        current_query=state["current_query"],
        tool_calls=state["tool_calls"],
        response=response,
        error=state["error"],
        call_stage=state["call_stage"]
    )


def update_error(state: AgentState, error: str) -> AgentState:
    """
    Update state with error message.
    
    Args:
        state: Current state
        error: Error message
        
    Returns:
        Updated state with error
    """
    return AgentState(
        messages=state["messages"],
        context=state["context"],
        current_query=state["current_query"],
        tool_calls=state["tool_calls"],
        response=state["response"],
        error=error,
        call_stage=state["call_stage"]
    )


def update_call_stage(state: AgentState, stage: str) -> AgentState:
    """
    Update call stage.
    
    Args:
        state: Current state
        stage: New call stage (greeting, main, closing)
        
    Returns:
        Updated state
    """
    return AgentState(
        messages=state["messages"],
        context=state["context"],
        current_query=state["current_query"],
        tool_calls=state["tool_calls"],
        response=state["response"],
        error=state["error"],
        call_stage=stage
    )


def add_tool_call(state: AgentState, tool_name: str) -> AgentState:
    """
    Add tool call to state tracking.
    
    Args:
        state: Current state
        tool_name: Name of tool called
        
    Returns:
        Updated state
    """
    new_tool_calls = list(state["tool_calls"]) + [tool_name]
    
    return AgentState(
        messages=state["messages"],
        context=state["context"],
        current_query=state["current_query"],
        tool_calls=new_tool_calls,
        response=state["response"],
        error=state["error"],
        call_stage=state["call_stage"]
    )


def add_user_message(state: AgentState, message: str) -> AgentState:
    """
    Add new user message to state.
    
    Args:
        state: Current state
        message: User's message
        
    Returns:
        Updated state with new message
    """
    new_messages = list(state["messages"]) + [HumanMessage(content=message)]
    
    return AgentState(
        messages=new_messages,
        context=None,  # Reset context for new query
        current_query=message,
        tool_calls=[],  # Reset tool calls for new turn
        response=None,  # Reset response
        error=None,  # Reset error
        call_stage=state["call_stage"]
    )


# =============================================================================
# STATE UTILITY FUNCTIONS
# =============================================================================

def get_message_history(state: AgentState) -> List[BaseMessage]:
    """
    Get message history from state.
    
    Args:
        state: Current state
        
    Returns:
        List of messages
    """
    return list(state["messages"])


def get_last_user_message(state: AgentState) -> Optional[str]:
    """
    Get the last user message from state.
    
    Args:
        state: Current state
        
    Returns:
        Last user message content or None
    """
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return None


def get_last_ai_message(state: AgentState) -> Optional[str]:
    """
    Get the last AI message from state.
    
    Args:
        state: Current state
        
    Returns:
        Last AI message content or None
    """
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return None


def get_conversation_length(state: AgentState) -> int:
    """
    Get number of messages in conversation.
    
    Args:
        state: Current state
        
    Returns:
        Number of messages
    """
    return len(state["messages"])


def has_error(state: AgentState) -> bool:
    """
    Check if state has an error.
    
    Args:
        state: Current state
        
    Returns:
        True if error exists
    """
    return state["error"] is not None


def has_context(state: AgentState) -> bool:
    """
    Check if state has retrieved context.
    
    Args:
        state: Current state
        
    Returns:
        True if context exists
    """
    return state["context"] is not None and len(state["context"]) > 0


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test state module."""
    
    print("=" * 70)
    print("Agent State Test (LangGraph Pattern)")
    print("=" * 70)
    
    # Test create_initial_state
    print("\n1. Testing create_initial_state():")
    print("-" * 50)
    
    state = create_initial_state(
        system_prompt="You are Sarah, a helpful assistant.",
        initial_message="Hello, I need to schedule an appointment."
    )
    
    print(f"   Messages: {len(state['messages'])}")
    print(f"   Call Stage: {state['call_stage']}")
    print(f"   Current Query: {state['current_query']}")
    
    # Test create_state_from_query
    print("\n2. Testing create_state_from_query():")
    print("-" * 50)
    
    state = create_state_from_query("What are your hours?")
    print(f"   Messages: {len(state['messages'])}")
    print(f"   Current Query: {state['current_query']}")
    
    # Test update functions
    print("\n3. Testing update functions:")
    print("-" * 50)
    
    state = update_context(state, "We are open 8am to 8pm daily.")
    print(f"   Context updated: {has_context(state)}")
    
    state = update_response(state, "We're open from 8am to 8pm every day!")
    print(f"   Response: {state['response'][:50]}...")
    
    state = update_call_stage(state, "main")
    print(f"   Call Stage: {state['call_stage']}")
    
    state = add_tool_call(state, "search_sop_tool")
    print(f"   Tool Calls: {state['tool_calls']}")
    
    # Test utility functions
    print("\n4. Testing utility functions:")
    print("-" * 50)
    
    print(f"   get_last_user_message(): {get_last_user_message(state)}")
    print(f"   get_last_ai_message(): {get_last_ai_message(state)[:50]}...")
    print(f"   get_conversation_length(): {get_conversation_length(state)}")
    print(f"   has_error(): {has_error(state)}")
    print(f"   has_context(): {has_context(state)}")
    
    # Test add_user_message
    print("\n5. Testing add_user_message():")
    print("-" * 50)
    
    state = add_user_message(state, "Can I also walk in?")
    print(f"   New query: {state['current_query']}")
    print(f"   Messages: {get_conversation_length(state)}")
    print(f"   Context reset: {not has_context(state)}")
    
    print("\n" + "=" * 70)
    print("State Module Ready")
    print("=" * 70)