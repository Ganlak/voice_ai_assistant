"""
Services Module
===============
Service layer for Voice AI POC.

Author: ThubIQ Healthcare
Version: 2.0.0
"""

# Existing services
from src.services.llm_service import llm_service
from src.services.stt_service import stt_service
from src.services.tts_service import tts_service

# New SOP service
from src.services.sop_service import (
    get_sop_service,
    get_retriever,
    get_vectorstore,
    search_sop,
    search_sop_with_scores,
    get_relevant_context,
    SOPService
)

__all__ = [
    # Existing
    "llm_service",
    "stt_service",
    "tts_service",
    # New SOP
    "get_sop_service",
    "get_retriever",
    "get_vectorstore",
    "search_sop",
    "search_sop_with_scores",
    "get_relevant_context",
    "SOPService"
]