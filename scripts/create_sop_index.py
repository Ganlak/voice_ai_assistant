"""
SOP FAISS Index Creator - MVP Production Grade (Multi-Document)
===============================================================
Indexes ALL .docx documents in the data folder using LangChain.

Features:
- Multi-document support
- Pure LangChain components
- Source tracking per document
- Robust error handling

Usage:
    python scripts/create_sop_index.py

Author: ThubIQ Healthcare
Version: 2.1.0
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure production-grade logging."""
    
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("sop_indexer")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"index_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration from environment variables."""
    
    REQUIRED_VARS = [
        "DATA_DIR",
        "FAISS_INDEX_DIR",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING"
    ]
    
    def __init__(self):
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "sop_faiss_index")
        self.chunk_size = int(os.getenv("SOP_CHUNK_SIZE", 500))
        self.chunk_overlap = int(os.getenv("SOP_CHUNK_OVERLAP", 100))
        self.top_k = int(os.getenv("SOP_RETRIEVAL_TOP_K", 3))
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-15-preview")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING")
    
    @property
    def data_path(self) -> Path:
        return project_root / self.data_dir
    
    @property
    def index_path(self) -> Path:
        return project_root / self.faiss_index_dir
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate required configuration."""
        missing = [var for var in self.REQUIRED_VARS if not os.getenv(var)]
        return len(missing) == 0, missing
    
    def to_dict(self) -> Dict:
        return {
            "data_dir": self.data_dir,
            "faiss_index_dir": self.faiss_index_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_deployment": self.embedding_deployment
        }


# =============================================================================
# DOCUMENT DISCOVERY
# =============================================================================

class DocumentDiscovery:
    """Discover all documents in data folder."""
    
    SUPPORTED_EXTENSIONS = [".docx", ".doc"]
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def find_documents(self) -> List[Path]:
        """Find all supported documents in data directory."""
        
        data_path = self.config.data_path
        
        if not data_path.exists():
            self.logger.error(f"Data directory not found: {data_path}")
            return []
        
        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            documents.extend(data_path.glob(f"*{ext}"))
        
        self.logger.info(f"Found {len(documents)} document(s) in {data_path}")
        
        for doc in documents:
            size_kb = doc.stat().st_size / 1024
            self.logger.info(f"   - {doc.name} ({size_kb:.1f} KB)")
        
        return documents


# =============================================================================
# DOCUMENT LOADER (Multi-Document)
# =============================================================================

class MultiDocumentLoader:
    """Load multiple documents using LangChain."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load(self, doc_paths: List[Path]) -> List[Document]:
        """Load all documents and combine."""
        
        all_documents = []
        
        for doc_path in doc_paths:
            self.logger.info(f"Loading: {doc_path.name}")
            
            try:
                # LangChain document loader
                loader = Docx2txtLoader(str(doc_path))
                documents = loader.load()
                
                # Add source metadata to each document
                for doc in documents:
                    doc.metadata["source"] = doc_path.name
                    doc.metadata["source_path"] = str(doc_path)
                
                all_documents.extend(documents)
                
                self.logger.info(f"   Loaded {len(documents)} section(s) from {doc_path.name}")
                
            except Exception as e:
                self.logger.error(f"   Failed to load {doc_path.name}: {str(e)}")
        
        self.logger.info(f"Total loaded: {len(all_documents)} document section(s)")
        
        return all_documents


# =============================================================================
# TEXT CHUNKER (LangChain)
# =============================================================================

class TextChunker:
    """Chunk documents using LangChain."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # LangChain text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                ""
            ]
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        
        self.logger.info(f"Chunking with size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        
        start_time = time.time()
        
        # LangChain chunking
        chunks = self.splitter.split_documents(documents)
        
        chunk_time = time.time() - start_time
        
        self.logger.info(f"Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Log chunks per source
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            self.logger.info(f"   - {source}: {count} chunks")
        
        return chunks


# =============================================================================
# METADATA ENRICHER
# =============================================================================

class MetadataEnricher:
    """Enrich chunks with metadata."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def enrich(self, chunks: List[Document]) -> List[Document]:
        """Add metadata to chunks."""
        
        self.logger.info("Enriching metadata...")
        
        total_chunks = len(chunks)
        enriched = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            
            # Calculate position
            position_pct = (i / total_chunks) * 100
            if position_pct < 20:
                position = "beginning"
            elif position_pct > 80:
                position = "end"
            else:
                position = "middle"
            
            # Create enriched document
            enriched_doc = Document(
                page_content=content,
                metadata={
                    "chunk_id": f"chunk_{i+1:03d}",
                    "index": i + 1,
                    "total_chunks": total_chunks,
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "position": position,
                    "source": chunk.metadata.get("source", "unknown"),
                    "indexed_at": datetime.now().isoformat()
                }
            )
            enriched.append(enriched_doc)
        
        self.logger.info(f"Enriched {len(enriched)} chunks")
        
        return enriched


# =============================================================================
# EMBEDDINGS SERVICE
# =============================================================================

class EmbeddingsService:
    """Initialize embeddings using LangChain."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.embeddings = None
    
    def initialize(self) -> bool:
        """Initialize embeddings with retry."""
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self.logger.info(f"Initializing embeddings (attempt {attempt}/{self.MAX_RETRIES})")
                
                # LangChain Azure OpenAI Embeddings
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=self.config.embedding_deployment,
                    azure_endpoint=self.config.azure_endpoint,
                    api_key=self.config.azure_api_key,
                    api_version=self.config.azure_api_version
                )
                
                # Test embeddings
                test_result = self.embeddings.embed_query("test")
                
                if test_result and len(test_result) > 0:
                    self.logger.info(f"Embeddings initialized. Dimension: {len(test_result)}")
                    return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed: {str(e)}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
        
        self.logger.error("Failed to initialize embeddings")
        return False
    
    def get_embeddings(self):
        return self.embeddings


# =============================================================================
# FAISS INDEX SERVICE
# =============================================================================

class FAISSIndexService:
    """Create and save FAISS index using LangChain."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.vectorstore = None
    
    def create(self, chunks: List[Document], embeddings) -> bool:
        """Create FAISS index."""
        
        self.logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
        
        start_time = time.time()
        
        try:
            # LangChain FAISS
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            create_time = time.time() - start_time
            self.logger.info(f"Index created in {create_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def save(self) -> bool:
        """Save index to disk."""
        
        if not self.vectorstore:
            self.logger.error("No index to save")
            return False
        
        try:
            output_dir = self.config.index_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.vectorstore.save_local(str(output_dir))
            
            self.logger.info(f"Index saved: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            return False
    
    def validate(self) -> bool:
        """Validate index files."""
        
        index_file = self.config.index_path / "index.faiss"
        pkl_file = self.config.index_path / "index.pkl"
        
        if not index_file.exists() or not pkl_file.exists():
            self.logger.error("Index files missing")
            return False
        
        self.logger.info("Index files validated")
        return True
    
    def get_vectorstore(self):
        return self.vectorstore


# =============================================================================
# INDEX INFO GENERATOR
# =============================================================================

class IndexInfoGenerator:
    """Generate index metadata."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def generate(self, chunks: List[Document], doc_paths: List[Path], metrics: Dict) -> Dict:
        """Generate index info JSON."""
        
        # Calculate stats
        char_counts = [c.metadata["char_count"] for c in chunks]
        word_counts = [c.metadata["word_count"] for c in chunks]
        
        # Source distribution
        source_dist = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_dist[source] = source_dist.get(source, 0) + 1
        
        index_info = {
            "meta": {
                "version": "2.1.0-MVP-MultiDoc",
                "created_at": datetime.now().isoformat(),
                "created_by": "SOP Index Creator"
            },
            "sources": [
                {
                    "name": p.name,
                    "path": str(p),
                    "chunks": source_dist.get(p.name, 0)
                }
                for p in doc_paths
            ],
            "settings": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": self.config.top_k,
                "embedding_model": self.config.embedding_deployment
            },
            "statistics": {
                "total_documents": len(doc_paths),
                "total_chunks": len(chunks),
                "total_characters": sum(char_counts),
                "total_words": sum(word_counts),
                "avg_chunk_chars": round(sum(char_counts) / len(char_counts), 1),
                "avg_chunk_words": round(sum(word_counts) / len(word_counts), 1)
            },
            "source_distribution": source_dist,
            "performance": metrics,
            "chunks": [
                {
                    "id": c.metadata["chunk_id"],
                    "source": c.metadata["source"],
                    "chars": c.metadata["char_count"],
                    "words": c.metadata["word_count"],
                    "position": c.metadata["position"],
                    "preview": c.page_content[:80].replace("\n", " ") + "..."
                }
                for c in chunks
            ]
        }
        
        return index_info
    
    def save(self, index_info: Dict) -> bool:
        """Save index info to JSON."""
        
        try:
            output_path = self.config.index_path / "index_info.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(index_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Index info saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index info: {str(e)}")
            return False


# =============================================================================
# RETRIEVAL TESTER
# =============================================================================

class RetrievalTester:
    """Test retrieval quality."""
    
    TEST_QUERIES = [
        "How should I greet a caller?",
        "What is the scheduling process?",
        "How do I handle cancellations?",
        "What is the late arrival policy?",
        "How should I end a call?",
        "What are the communication guidelines?",
        "How do I handle walk-ins?",
        "What should I say about wait times?",
        "How do I provide directions?",
        "What is the online booking process?"
    ]
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def run_tests(self, vectorstore) -> Dict:
        """Run retrieval tests."""
        
        self.logger.info("Running retrieval tests...")
        
        results = []
        total_time = 0
        
        for query in self.TEST_QUERIES:
            start = time.time()
            
            search_results = vectorstore.similarity_search_with_score(
                query,
                k=self.config.top_k
            )
            
            query_time = time.time() - start
            total_time += query_time
            
            results.append({
                "query": query,
                "time_ms": round(query_time * 1000, 2),
                "results": [
                    {
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "source": doc.metadata.get("source"),
                        "score": round(float(score), 4),
                        "preview": doc.page_content[:100] + "..."
                    }
                    for doc, score in search_results
                ]
            })
        
        metrics = {
            "total_queries": len(self.TEST_QUERIES),
            "total_time_ms": round(total_time * 1000, 2),
            "avg_time_ms": round((total_time / len(self.TEST_QUERIES)) * 1000, 2)
        }
        
        self.logger.info(f"Tests complete. Avg time: {metrics['avg_time_ms']}ms")
        
        return metrics, results
    
    def print_results(self, results: List[Dict]):
        """Print test results."""
        
        print("\n" + "=" * 70)
        print("RETRIEVAL TEST RESULTS")
        print("=" * 70)
        
        for result in results:
            print(f"\nQuery: \"{result['query']}\"")
            print(f"Time: {result['time_ms']}ms")
            print("-" * 60)
            
            for i, r in enumerate(result["results"]):
                print(f"   [{i+1}] Score: {r['score']:.4f} | Source: {r['source']}")
                print(f"       Chunk: {r['chunk_id']}")
                print(f"       {r['preview']}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class SOPIndexer:
    """Main indexing orchestrator."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.config = Config()
    
    def run(self) -> bool:
        """Execute indexing pipeline."""
        
        pipeline_start = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("SOP FAISS Index Creator - MVP (Multi-Document)")
        self.logger.info("=" * 70)
        
        # Step 0: Validate config
        self.logger.info("Step 0: Validating Configuration")
        valid, missing = self.config.validate()
        if not valid:
            self.logger.error(f"Missing environment variables: {missing}")
            return False
        
        self.logger.info(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")
        
        # Step 1: Discover documents
        self.logger.info("Step 1: Discovering Documents")
        discovery = DocumentDiscovery(self.config, self.logger)
        doc_paths = discovery.find_documents()
        
        if not doc_paths:
            self.logger.error("No documents found in data directory")
            return False
        
        # Step 2: Load documents
        self.logger.info("Step 2: Loading Documents")
        loader = MultiDocumentLoader(self.config, self.logger)
        documents = loader.load(doc_paths)
        
        if not documents:
            self.logger.error("No documents loaded")
            return False
        
        # Step 3: Chunk documents
        self.logger.info("Step 3: Chunking Documents")
        chunker = TextChunker(self.config, self.logger)
        chunks = chunker.chunk(documents)
        
        # Step 4: Enrich metadata
        self.logger.info("Step 4: Enriching Metadata")
        enricher = MetadataEnricher(self.config, self.logger)
        enriched_chunks = enricher.enrich(chunks)
        
        # Step 5: Initialize embeddings
        self.logger.info("Step 5: Initializing Embeddings")
        embeddings_service = EmbeddingsService(self.config, self.logger)
        if not embeddings_service.initialize():
            return False
        
        # Step 6: Create FAISS index
        self.logger.info("Step 6: Creating FAISS Index")
        index_service = FAISSIndexService(self.config, self.logger)
        if not index_service.create(enriched_chunks, embeddings_service.get_embeddings()):
            return False
        
        # Step 7: Save index
        self.logger.info("Step 7: Saving Index")
        if not index_service.save():
            return False
        
        # Step 8: Validate index
        self.logger.info("Step 8: Validating Index")
        if not index_service.validate():
            return False
        
        # Step 9: Test retrieval
        self.logger.info("Step 9: Testing Retrieval")
        tester = RetrievalTester(self.config, self.logger)
        metrics, results = tester.run_tests(index_service.get_vectorstore())
        tester.print_results(results)
        
        # Step 10: Save index info
        self.logger.info("Step 10: Saving Index Info")
        pipeline_time = time.time() - pipeline_start
        
        performance_metrics = {
            "pipeline_time_seconds": round(pipeline_time, 2),
            "retrieval_avg_ms": metrics["avg_time_ms"]
        }
        
        info_generator = IndexInfoGenerator(self.config, self.logger)
        index_info = info_generator.generate(enriched_chunks, doc_paths, performance_metrics)
        info_generator.save(index_info)
        
        # Final summary
        self.logger.info("=" * 70)
        self.logger.info("MVP INDEX CREATED SUCCESSFULLY")
        self.logger.info("=" * 70)
        self.logger.info(f"   Documents: {len(doc_paths)}")
        for p in doc_paths:
            self.logger.info(f"      - {p.name}")
        self.logger.info(f"   Total Chunks: {len(enriched_chunks)}")
        self.logger.info(f"   Index Path: {self.config.index_path}")
        self.logger.info(f"   Total Time: {pipeline_time:.2f}s")
        self.logger.info("=" * 70)
        
        return True


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    indexer = SOPIndexer()
    success = indexer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()