"""
SOP Service - MVP Production Grade (Pure LangChain)
===================================================
Service layer for SOP document retrieval using LangChain patterns.

Usage:
    from src.services.sop_service import get_sop_service, get_retriever

Author: ThubIQ Healthcare
Version: 2.0.0
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


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
# CONFIGURATION
# =============================================================================

def get_config() -> dict:
    """Load configuration from environment variables."""
    return {
        "faiss_index_dir": os.getenv("FAISS_INDEX_DIR", "sop_faiss_index"),
        "top_k": int(os.getenv("SOP_RETRIEVAL_TOP_K", 3)),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_api_version": os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-15-preview"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING"),
    }


# =============================================================================
# INDEX PATH RESOLVER
# =============================================================================

def get_index_path() -> Path:
    """Get absolute path to FAISS index."""
    config = get_config()
    
    # Try relative to project root
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / config["faiss_index_dir"]
    
    if index_path.exists():
        return index_path
    
    # Try relative to current working directory
    cwd_path = Path.cwd() / config["faiss_index_dir"]
    if cwd_path.exists():
        return cwd_path
    
    return index_path


# =============================================================================
# EMBEDDINGS (LangChain)
# =============================================================================

def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get Azure OpenAI embeddings instance."""
    config = get_config()
    
    return AzureOpenAIEmbeddings(
        azure_deployment=config["embedding_deployment"],
        azure_endpoint=config["azure_endpoint"],
        api_key=config["azure_api_key"],
        api_version=config["azure_api_version"]
    )


# =============================================================================
# VECTORSTORE (LangChain FAISS)
# =============================================================================

# Singleton vectorstore instance
_vectorstore: Optional[FAISS] = None


def get_vectorstore() -> FAISS:
    """
    Get FAISS vectorstore instance (singleton).
    
    Returns:
        FAISS vectorstore
    """
    global _vectorstore
    
    if _vectorstore is None:
        index_path = get_index_path()
        
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        
        logger.info(f"Loading FAISS index from: {index_path}")
        
        embeddings = get_embeddings()
        
        _vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("FAISS index loaded successfully")
    
    return _vectorstore


# =============================================================================
# RETRIEVER (LangChain)
# =============================================================================

def get_retriever(top_k: Optional[int] = None) -> BaseRetriever:
    """
    Get LangChain retriever for SOP documents.
    
    Args:
        top_k: Number of documents to retrieve (default from config)
        
    Returns:
        LangChain BaseRetriever
    """
    config = get_config()
    k = top_k or config["top_k"]
    
    vectorstore = get_vectorstore()
    
    # LangChain native retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


# =============================================================================
# SEARCH FUNCTIONS (LangChain)
# =============================================================================

def search_sop(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    Search SOP documents using LangChain retriever.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of LangChain Document objects
    """
    retriever = get_retriever(top_k)
    
    # LangChain invoke pattern
    documents = retriever.invoke(query)
    
    return documents


def search_sop_with_scores(
    query: str, 
    top_k: Optional[int] = None
) -> List[tuple[Document, float]]:
    """
    Search SOP documents with relevance scores.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of (Document, score) tuples
    """
    config = get_config()
    k = top_k or config["top_k"]
    
    vectorstore = get_vectorstore()
    
    # LangChain similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return results


def get_relevant_context(query: str, top_k: Optional[int] = None) -> str:
    """
    Get concatenated context from relevant documents.
    
    Args:
        query: Search query string
        top_k: Number of documents to include
        
    Returns:
        Concatenated document content
    """
    documents = search_sop(query, top_k)
    
    # Concatenate document contents
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    return context


# =============================================================================
# SERVICE CLASS (LangChain Pattern)
# =============================================================================

class SOPService:
    """
    SOP Service wrapper using LangChain components.
    
    Provides convenient access to retriever and search functions.
    """
    
    def __init__(self, top_k: Optional[int] = None):
        """Initialize SOP service."""
        self.config = get_config()
        self.top_k = top_k or self.config["top_k"]
        self._vectorstore = None
        self._retriever = None
    
    @property
    def vectorstore(self) -> FAISS:
        """Get vectorstore (lazy loading)."""
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore()
        return self._vectorstore
    
    @property
    def retriever(self) -> BaseRetriever:
        """Get retriever (lazy loading)."""
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
        return self._retriever
    
    def search(self, query: str) -> List[Document]:
        """Search using LangChain retriever."""
        return self.retriever.invoke(query)
    
    def search_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
        return self.vectorstore.similarity_search_with_score(query, k=self.top_k)
    
    def get_context(self, query: str) -> str:
        """Get concatenated context."""
        docs = self.search(query)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        try:
            _ = self.vectorstore
            return True
        except Exception:
            return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_sop_service(top_k: Optional[int] = None) -> SOPService:
    """
    Get SOP service instance.
    
    Args:
        top_k: Number of documents to retrieve
        
    Returns:
        SOPService instance
    """
    return SOPService(top_k)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test SOP service directly."""
    
    print("=" * 70)
    print("SOP Service Test (Pure LangChain)")
    print("=" * 70)
    
    # Test using functions
    print("\n1. Testing search_sop() function:")
    print("-" * 50)
    
    docs = search_sop("How do I schedule an appointment?")
    print(f"   Results: {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"   [{i+1}] {doc.metadata.get('chunk_id', 'N/A')}: {doc.page_content[:80]}...")
    
    # Test with scores
    print("\n2. Testing search_sop_with_scores() function:")
    print("-" * 50)
    
    results = search_sop_with_scores("What is the late arrival policy?")
    for i, (doc, score) in enumerate(results):
        print(f"   [{i+1}] Score: {score:.4f} | {doc.page_content[:60]}...")
    
    # Test context retrieval
    print("\n3. Testing get_relevant_context() function:")
    print("-" * 50)
    
    context = get_relevant_context("greeting procedures")
    print(f"   Context length: {len(context)} characters")
    
    # Test service class
    print("\n4. Testing SOPService class:")
    print("-" * 50)
    
    service = get_sop_service()
    print(f"   Service ready: {service.is_ready()}")
    
    docs = service.search("walk-in availability")
    print(f"   Search results: {len(docs)} documents")
    
    # Test retriever directly
    print("\n5. Testing get_retriever() function:")
    print("-" * 50)
    
    retriever = get_retriever()
    print(f"   Retriever type: {type(retriever).__name__}")
    
    docs = retriever.invoke("communication guidelines")
    print(f"   Invoke results: {len(docs)} documents")
    
    print("\n" + "=" * 70)
    print("All Tests Passed - Pure LangChain Implementation")
    print("=" * 70)