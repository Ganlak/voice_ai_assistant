"""
SOP Search Tool - MVP Production Grade (Pure LangChain)
=======================================================
LangChain tool for searching SOP documents.
Designed for LangGraph agent integration.

Usage:
    from src.tools.search_sop import search_sop_tool

Author: ThubIQ Healthcare
Version: 2.0.0
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_core.tools import tool

# Direct import from service module (avoid circular import via src package)
from src.services.sop_service import search_sop


# =============================================================================
# LANGCHAIN TOOL DEFINITION
# =============================================================================

@tool
def search_sop_tool(query: str) -> str:
    """
    Search the SOP (Standard Operating Procedures) document for relevant information.
    
    Use this tool when you need to find procedures, guidelines, or policies for:
    - Greeting callers
    - Scheduling, rescheduling, or cancelling appointments
    - Handling walk-in requests
    - Providing directions to the clinic
    - Explaining wait times
    - Handling late arrivals
    - Online booking assistance
    - Communication guidelines
    - Call closing procedures
    
    Args:
        query: The search query describing what information you need.
               Be specific about the topic (e.g., "appointment scheduling process",
               "late arrival policy", "greeting procedures").
    
    Returns:
        Relevant SOP content that can be used to answer the caller's question.
    """
    
    # Get documents using LangChain retriever
    documents = search_sop(query)
    
    if not documents:
        return "No relevant procedures found for this query."
    
    # Format results for LLM consumption
    results = []
    for i, doc in enumerate(documents, 1):
        chunk_id = doc.metadata.get("chunk_id", f"result_{i}")
        content = doc.page_content.strip()
        results.append(f"[{chunk_id}]\n{content}")
    
    return "\n\n---\n\n".join(results)


@tool
def get_procedure(topic: str) -> str:
    """
    Get specific procedure or guideline from the SOP document.
    
    Use this tool to retrieve detailed procedures for specific topics:
    - "greeting" - How to greet callers
    - "scheduling" - Appointment scheduling process
    - "cancellation" - How to handle cancellations
    - "reschedule" - Rescheduling appointments
    - "walk-in" - Walk-in availability and policies
    - "directions" - How to provide clinic directions
    - "wait times" - What to say about wait times
    - "late arrival" - 15-minute late arrival policy
    - "online booking" - Website scheduling assistance
    - "closing" - How to end calls properly
    - "hold" - Putting callers on hold
    - "communication" - General communication guidelines
    
    Args:
        topic: The specific topic or procedure name.
    
    Returns:
        Detailed procedure information for the specified topic.
    """
    
    # Map common topics to better search queries
    topic_queries = {
        "greeting": "greeting caller opening phone answer",
        "scheduling": "schedule new appointment booking",
        "cancellation": "cancel appointment cancellation",
        "reschedule": "reschedule change appointment",
        "walk-in": "walk-in without appointment availability",
        "directions": "directions location clinic address",
        "wait times": "wait time current delay how long",
        "late arrival": "late arrival 15 minute policy running late",
        "online booking": "online website scheduling book appointment",
        "closing": "closing end call goodbye hang up",
        "hold": "hold waiting please hold",
        "communication": "communication guidelines tone professional"
    }
    
    # Use mapped query or original topic
    search_query = topic_queries.get(topic.lower(), topic)
    
    # Get documents using LangChain retriever
    documents = search_sop(search_query)
    
    if not documents:
        return f"No procedure found for topic: {topic}"
    
    # Return top result with context
    top_doc = documents[0]
    content = top_doc.page_content.strip()
    
    return f"Procedure for '{topic}':\n\n{content}"


# =============================================================================
# TOOL LIST FOR AGENT
# =============================================================================

# Export tools list for LangGraph agent
sop_tools = [search_sop_tool, get_procedure]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test SOP tools directly."""
    
    print("=" * 70)
    print("SOP Tools Test (Pure LangChain)")
    print("=" * 70)
    
    # Test search_sop_tool
    print("\n1. Testing search_sop_tool:")
    print("-" * 50)
    
    result = search_sop_tool.invoke({"query": "How do I schedule an appointment?"})
    print(f"Query: 'How do I schedule an appointment?'")
    print(f"Result:\n{result[:500]}...")
    
    # Test get_procedure
    print("\n2. Testing get_procedure:")
    print("-" * 50)
    
    topics = ["greeting", "late arrival", "walk-in"]
    
    for topic in topics:
        result = get_procedure.invoke({"topic": topic})
        print(f"\nTopic: '{topic}'")
        print(f"Result:\n{result[:300]}...")
    
    # Test tool metadata
    print("\n3. Tool Metadata:")
    print("-" * 50)
    
    print(f"search_sop_tool.name: {search_sop_tool.name}")
    print(f"search_sop_tool.description: {search_sop_tool.description[:100]}...")
    
    print(f"\nget_procedure.name: {get_procedure.name}")
    print(f"get_procedure.description: {get_procedure.description[:100]}...")
    
    # Test tools list
    print("\n4. Tools List:")
    print("-" * 50)
    
    print(f"Total tools: {len(sop_tools)}")
    for t in sop_tools:
        print(f"   - {t.name}")
    
    print("\n" + "=" * 70)
    print("All Tools Tests Passed")
    print("=" * 70)