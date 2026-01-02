"""
SOP Retrieval Test Script - MVP Production Grade
================================================
Tests FAISS index retrieval with comprehensive query validation.
Supports batch testing and interactive mode.

Usage:
    python scripts/test_sop_retrieval.py
    python scripts/test_sop_retrieval.py --interactive

Author: ThubIQ Healthcare
Version: 2.0.0
"""

import os
import sys
import json
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure production-grade logging."""
    
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("sop_retrieval_test")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"retrieval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
    
    def __init__(self):
        self.faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "sop_faiss_index")
        self.top_k = int(os.getenv("SOP_RETRIEVAL_TOP_K", 3))
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-15-preview")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING")
    
    @property
    def index_path(self) -> Path:
        return project_root / self.faiss_index_dir
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate required configuration."""
        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_EMBEDDING"]
        missing = [var for var in required if not os.getenv(var)]
        return len(missing) == 0, missing


# =============================================================================
# INDEX LOADER
# =============================================================================

class IndexLoader:
    """Load and validate FAISS index."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.vectorstore = None
        self.index_info = None
    
    def validate_index_files(self) -> bool:
        """Check if index files exist."""
        
        index_path = self.config.index_path
        
        required_files = ["index.faiss", "index.pkl"]
        
        for file in required_files:
            if not (index_path / file).exists():
                self.logger.error(f"Missing index file: {file}")
                return False
        
        self.logger.info(f"Index files validated: {index_path}")
        return True
    
    def load_index_info(self) -> Optional[Dict]:
        """Load index metadata."""
        
        info_path = self.config.index_path / "index_info.json"
        
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self.index_info = json.load(f)
            self.logger.info(f"Index info loaded: {self.index_info.get('statistics', {}).get('total_chunks', 'N/A')} chunks")
            return self.index_info
        
        self.logger.warning("index_info.json not found")
        return None
    
    def load(self) -> Optional[FAISS]:
        """Load FAISS vectorstore."""
        
        if not self.validate_index_files():
            return None
        
        self.logger.info("Loading FAISS index...")
        
        start_time = time.time()
        
        try:
            # Initialize embeddings
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.config.embedding_deployment,
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version
            )
            
            # Load vectorstore
            self.vectorstore = FAISS.load_local(
                str(self.config.index_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            load_time = time.time() - start_time
            
            self.logger.info(f"Index loaded in {load_time:.2f}s")
            
            # Load index info
            self.load_index_info()
            
            return self.vectorstore
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            return None
    
    def get_vectorstore(self) -> Optional[FAISS]:
        """Return loaded vectorstore."""
        return self.vectorstore
    
    def get_index_info(self) -> Optional[Dict]:
        """Return index metadata."""
        return self.index_info


# =============================================================================
# RETRIEVAL SERVICE
# =============================================================================

class RetrievalService:
    """Handle retrieval operations."""
    
    def __init__(self, vectorstore: FAISS, config: Config, logger: logging.Logger):
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logger
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple]:
        """Perform similarity search."""
        
        k = top_k or self.config.top_k
        
        start_time = time.time()
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        search_time = time.time() - start_time
        
        self.logger.debug(f"Query: '{query[:50]}...' | Results: {len(results)} | Time: {search_time*1000:.2f}ms")
        
        return results, search_time
    
    def search_with_filter(self, query: str, filter_dict: Dict, top_k: Optional[int] = None) -> List[Tuple]:
        """Perform filtered similarity search."""
        
        k = top_k or self.config.top_k
        
        start_time = time.time()
        
        # Note: FAISS filtering depends on metadata
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        search_time = time.time() - start_time
        
        return results, search_time
    
    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Get concatenated relevant context for LLM."""
        
        results, _ = self.search(query, top_k)
        
        context_parts = []
        for doc, score in results:
            context_parts.append(doc.page_content)
        
        return "\n\n---\n\n".join(context_parts)


# =============================================================================
# BATCH TESTER
# =============================================================================

class BatchTester:
    """Run batch retrieval tests."""
    
    # Comprehensive test queries covering all SOP topics
    TEST_QUERIES = [
        # Greeting
        {"query": "How should I greet a caller?", "expected_topic": "greeting"},
        {"query": "What do I say when answering the phone?", "expected_topic": "greeting"},
        
        # Communication
        {"query": "What are the communication guidelines?", "expected_topic": "communication"},
        {"query": "How should I handle putting someone on hold?", "expected_topic": "communication"},
        {"query": "What do I do during silence or wait times?", "expected_topic": "communication"},
        
        # Closing
        {"query": "How should I end a call?", "expected_topic": "closing"},
        {"query": "What do I say before hanging up?", "expected_topic": "closing"},
        
        # Appointment Scheduling
        {"query": "How do I schedule an appointment?", "expected_topic": "scheduling"},
        {"query": "What is the process for booking?", "expected_topic": "scheduling"},
        {"query": "How do I handle cancellations?", "expected_topic": "scheduling"},
        {"query": "How do I reschedule an appointment?", "expected_topic": "scheduling"},
        
        # Walk-in
        {"query": "Do you accept walk-ins?", "expected_topic": "walk_in"},
        {"query": "Can someone come without an appointment?", "expected_topic": "walk_in"},
        
        # Directions
        {"query": "How do I give directions to the clinic?", "expected_topic": "directions"},
        {"query": "Where is the clinic located?", "expected_topic": "directions"},
        
        # Wait Times
        {"query": "What should I say about wait times?", "expected_topic": "wait_times"},
        {"query": "How long is the current wait?", "expected_topic": "wait_times"},
        
        # Online Scheduling
        {"query": "How does online booking work?", "expected_topic": "online"},
        {"query": "What is the website for scheduling?", "expected_topic": "online"},
        
        # Late Arrival
        {"query": "What is the policy for late arrivals?", "expected_topic": "late_arrival"},
        {"query": "What if someone is running late?", "expected_topic": "late_arrival"},
    ]
    
    def __init__(self, retrieval_service: RetrievalService, config: Config, logger: logging.Logger):
        self.retrieval_service = retrieval_service
        self.config = config
        self.logger = logger
        self.results = []
    
    def run(self) -> Dict:
        """Run all batch tests."""
        
        self.logger.info(f"Running {len(self.TEST_QUERIES)} test queries...")
        
        self.results = []
        total_time = 0
        
        for test in self.TEST_QUERIES:
            query = test["query"]
            expected = test["expected_topic"]
            
            results, search_time = self.retrieval_service.search(query)
            total_time += search_time
            
            self.results.append({
                "query": query,
                "expected_topic": expected,
                "time_ms": round(search_time * 1000, 2),
                "results": [
                    {
                        "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                        "score": round(score, 4),
                        "position": doc.metadata.get("position", "N/A"),
                        "preview": doc.page_content[:120].replace("\n", " ") + "..."
                    }
                    for doc, score in results
                ]
            })
        
        # Calculate metrics
        metrics = {
            "total_queries": len(self.TEST_QUERIES),
            "total_time_ms": round(total_time * 1000, 2),
            "avg_time_ms": round((total_time / len(self.TEST_QUERIES)) * 1000, 2),
            "min_time_ms": round(min(r["time_ms"] for r in self.results), 2),
            "max_time_ms": round(max(r["time_ms"] for r in self.results), 2),
            "avg_top_score": round(sum(r["results"][0]["score"] for r in self.results) / len(self.results), 4),
            "results": self.results
        }
        
        self.logger.info(f"Batch tests complete. Avg time: {metrics['avg_time_ms']}ms")
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print formatted test results."""
        
        print("\n" + "=" * 80)
        print("SOP RETRIEVAL TEST RESULTS")
        print("=" * 80)
        
        for result in metrics["results"]:
            print(f"\nQuery: \"{result['query']}\"")
            print(f"Expected Topic: {result['expected_topic']} | Time: {result['time_ms']}ms")
            print("-" * 70)
            
            for i, r in enumerate(result["results"]):
                print(f"   [{i+1}] Score: {r['score']:.4f} | Chunk: {r['chunk_id']} | Pos: {r['position']}")
                print(f"       {r['preview']}")
        
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"   Total Queries:    {metrics['total_queries']}")
        print(f"   Avg Query Time:   {metrics['avg_time_ms']}ms")
        print(f"   Min Query Time:   {metrics['min_time_ms']}ms")
        print(f"   Max Query Time:   {metrics['max_time_ms']}ms")
        print(f"   Avg Top Score:    {metrics['avg_top_score']}")
        print("=" * 80)
    
    def save_results(self, metrics: Dict, output_path: Path):
        """Save test results to JSON."""
        
        # Convert numpy float32 to Python float for JSON serialization
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj
        
        output = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "top_k": self.config.top_k
            },
            "metrics": {
                "total_queries": metrics["total_queries"],
                "avg_time_ms": float(metrics["avg_time_ms"]),
                "min_time_ms": float(metrics["min_time_ms"]),
                "max_time_ms": float(metrics["max_time_ms"]),
                "avg_top_score": float(metrics["avg_top_score"])
            },
            "results": convert_floats(metrics["results"])
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved: {output_path}")


# =============================================================================
# INTERACTIVE TESTER
# =============================================================================

class InteractiveTester:
    """Interactive query testing mode."""
    
    def __init__(self, retrieval_service: RetrievalService, config: Config, logger: logging.Logger):
        self.retrieval_service = retrieval_service
        self.config = config
        self.logger = logger
    
    def run(self):
        """Start interactive mode."""
        
        print("\n" + "=" * 80)
        print("INTERACTIVE RETRIEVAL MODE")
        print("=" * 80)
        print("Commands:")
        print("   <query>     - Search for relevant SOP content")
        print("   /top <n>    - Set number of results (e.g., /top 5)")
        print("   /context    - Show last query as LLM context")
        print("   /help       - Show this help")
        print("   /exit       - Exit interactive mode")
        print("=" * 80)
        
        top_k = self.config.top_k
        last_query = None
        
        while True:
            try:
                user_input = input("\nQuery > ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    
                    if cmd == "/exit" or cmd == "/quit" or cmd == "/q":
                        print("Exiting interactive mode.")
                        break
                    
                    elif cmd == "/help":
                        print("Commands: /top <n>, /context, /exit, /help")
                        continue
                    
                    elif cmd.startswith("/top "):
                        try:
                            top_k = int(cmd.split()[1])
                            print(f"Top K set to: {top_k}")
                        except:
                            print("Usage: /top <number>")
                        continue
                    
                    elif cmd == "/context":
                        if last_query:
                            context = self.retrieval_service.get_relevant_context(last_query, top_k)
                            print("\n--- LLM CONTEXT ---")
                            print(context)
                            print("--- END CONTEXT ---")
                        else:
                            print("No previous query. Enter a query first.")
                        continue
                    
                    else:
                        print(f"Unknown command: {cmd}")
                        continue
                
                # Perform search
                last_query = user_input
                results, search_time = self.retrieval_service.search(user_input, top_k)
                
                print(f"\nResults ({len(results)}) | Time: {search_time*1000:.2f}ms")
                print("-" * 70)
                
                for i, (doc, score) in enumerate(results):
                    chunk_id = doc.metadata.get("chunk_id", "N/A")
                    position = doc.metadata.get("position", "N/A")
                    preview = doc.page_content[:200].replace("\n", " ")
                    
                    print(f"\n[{i+1}] Score: {score:.4f} | Chunk: {chunk_id} | Position: {position}")
                    print(f"    {preview}...")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                self.logger.error(f"Error: {str(e)}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class RetrievalTestRunner:
    """Main test runner orchestrator."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.config = Config()
    
    def run(self, interactive: bool = False, save_results: bool = True) -> bool:
        """Execute test pipeline."""
        
        self.logger.info("=" * 70)
        self.logger.info("SOP Retrieval Test - MVP Production")
        self.logger.info("=" * 70)
        
        # Validate configuration
        valid, missing = self.config.validate()
        if not valid:
            self.logger.error(f"Missing environment variables: {missing}")
            return False
        
        # Load index
        self.logger.info("Loading FAISS index...")
        loader = IndexLoader(self.config, self.logger)
        vectorstore = loader.load()
        
        if not vectorstore:
            self.logger.error("Failed to load index")
            return False
        
        # Print index info
        index_info = loader.get_index_info()
        if index_info:
            stats = index_info.get("statistics", {})
            print(f"\nIndex Statistics:")
            print(f"   Chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"   Avg Chunk Size: {stats.get('avg_chunk_chars', 'N/A')} chars")
        
        # Initialize retrieval service
        retrieval_service = RetrievalService(vectorstore, self.config, self.logger)
        
        if interactive:
            # Interactive mode
            tester = InteractiveTester(retrieval_service, self.config, self.logger)
            tester.run()
        else:
            # Batch test mode
            self.logger.info("Running batch tests...")
            tester = BatchTester(retrieval_service, self.config, self.logger)
            metrics = tester.run()
            tester.print_results(metrics)
            
            # Save results
            if save_results:
                output_path = project_root / "logs" / f"retrieval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                tester.save_results(metrics, output_path)
        
        self.logger.info("Test run complete")
        return True


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="SOP Retrieval Test Script")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    runner = RetrievalTestRunner()
    success = runner.run(
        interactive=args.interactive,
        save_results=not args.no_save
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()