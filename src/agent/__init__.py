"""
Agent Module
============
LangGraph agent for Voice AI.

Author: ThubIQ Healthcare
Version: 2.0.0
"""

# Temporary placeholders to break circular import
# Will be restored after all modules are updated

def run_agent(query: str) -> str:
    """Placeholder - import actual function when needed."""
    from src.agent.graph import run_agent as _run_agent
    return _run_agent(query)

def agent(query: str) -> str:
    """Placeholder - import actual function when needed."""
    from src.agent.graph import agent as _agent
    return _agent(query)

def build_agent_graph():
    """Placeholder - import actual function when needed."""
    from src.agent.graph import build_agent_graph as _build
    return _build()

# Lazy imports for state and prompts
def __getattr__(name):
    if name == "AgentState":
        from src.agent.state import AgentState
        return AgentState
    if name == "create_initial_state":
        from src.agent.state import create_initial_state
        return create_initial_state
    if name == "update_state":
        from src.agent.state import update_state
        return update_state
    if name in ["SYSTEM_PROMPT", "VOICE_RESPONSE_PROMPT", "TOOL_SELECTION_PROMPT", 
                "ERROR_RESPONSE_PROMPT", "NO_RESULTS_PROMPT", "CLARIFICATION_PROMPT"]:
        from src.agent import prompts
        return getattr(prompts, name)
    raise AttributeError(f"module 'src.agent' has no attribute '{name}'")

__all__ = [
    "agent",
    "run_agent",
    "build_agent_graph",
    "AgentState",
    "create_initial_state",
    "update_state",
    "SYSTEM_PROMPT",
    "VOICE_RESPONSE_PROMPT",
    "TOOL_SELECTION_PROMPT",
    "ERROR_RESPONSE_PROMPT",
    "NO_RESULTS_PROMPT",
    "CLARIFICATION_PROMPT"
]