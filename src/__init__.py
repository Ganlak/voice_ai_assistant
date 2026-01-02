"""
Voice AI POC - SOP Based
========================
Voice AI assistant using SOP document for healthcare call center.

Author: ThubIQ Healthcare
Version: 2.0.0
"""

__version__ = "2.0.0"

from src.config.settings import (
    BASE_DIR,
    SERVER_HOST,
    SERVER_PORT
)

# Lazy imports to avoid circular dependencies
# These will be imported when accessed, not at module load time

def __getattr__(name):
    """Lazy import handler to avoid circular dependencies."""
    
    if name == "run_agent":
        from src.agent import run_agent
        return run_agent
    
    if name == "agent":
        from src.agent import agent
        return agent
    
    if name == "voice_call_handler":
        from src.voice import voice_call_handler
        return voice_call_handler
    
    if name == "simple_voice_handler":
        from src.voice import simple_voice_handler
        return simple_voice_handler
    
    if name == "llm_service":
        from src.services.llm_service import llm_service
        return llm_service
    
    if name == "stt_service":
        from src.services.stt_service import stt_service
        return stt_service
    
    if name == "tts_service":
        from src.services.tts_service import tts_service
        return tts_service
    
    if name == "TOOL_FUNCTIONS":
        from src.tools import TOOL_FUNCTIONS
        return TOOL_FUNCTIONS
    
    if name == "TOOL_DEFINITIONS":
        from src.tools import TOOL_DEFINITIONS
        return TOOL_DEFINITIONS
    
    raise AttributeError(f"module 'src' has no attribute '{name}'")


__all__ = [
    "__version__",
    "BASE_DIR",
    "SERVER_HOST",
    "SERVER_PORT",
    "run_agent",
    "agent",
    "voice_call_handler",
    "simple_voice_handler",
    "llm_service",
    "stt_service",
    "tts_service",
    "TOOL_FUNCTIONS",
    "TOOL_DEFINITIONS"
]