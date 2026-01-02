"""
SOP Tools Module
================
LangChain tools for SOP document retrieval.

Author: ThubIQ Healthcare
Version: 2.0.0
"""

from src.tools.search_sop import (
    search_sop_tool,
    get_procedure,
    sop_tools
)

# Tool list for LangGraph agent
TOOLS = sop_tools

# Backward compatibility
TOOL_FUNCTIONS = {
    "search_sop": search_sop_tool,
    "get_procedure": get_procedure
}

TOOL_DEFINITIONS = [
    {
        "name": "search_sop",
        "description": search_sop_tool.description
    },
    {
        "name": "get_procedure",
        "description": get_procedure.description
    }
]

__all__ = [
    "search_sop_tool",
    "get_procedure",
    "sop_tools",
    "TOOLS",
    "TOOL_FUNCTIONS",
    "TOOL_DEFINITIONS"
]