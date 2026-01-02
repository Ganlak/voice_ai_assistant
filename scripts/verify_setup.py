"""
ThubIQ Healthcare - Setup Verification Script
Checks all files, imports, and connections.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("  ThubIQ Healthcare - Setup Verification")
print("=" * 70)

# ========== STEP 1: Check File Structure ==========
print("\n📁 STEP 1: Checking File Structure")
print("-" * 50)

REQUIRED_FILES = [
    # Root files
    ".env",
    "bot.py",
    "server.py",
    
    # Config
    "src/__init__.py",
    "src/config/__init__.py",
    "src/config/settings.py",
    
    # Utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    
    # Services
    "src/services/__init__.py",
    "src/services/faiss_service.py",
    "src/services/llm_service.py",
    "src/services/stt_service.py",
    "src/services/tts_service.py",
    
    # Tools
    "src/tools/__init__.py",
    "src/tools/search_calls.py",
    "src/tools/get_call_details.py",
    "src/tools/filter_calls.py",
    "src/tools/summarize.py",
    
    # Agent
    "src/agent/__init__.py",
    "src/agent/prompts.py",
    "src/agent/state.py",
    "src/agent/graph.py",
    
    # Voice
    "src/voice/__init__.py",
    "src/voice/pipeline.py",
    "src/voice/handlers.py",
    
    # FAISS Index
    "mini_faiss_index/index.faiss",
    "mini_faiss_index/metadata.pkl",
    "mini_faiss_index/metadata.json",
    "mini_faiss_index/index_info.json",
]

missing_files = []
existing_files = []

for file_path in REQUIRED_FILES:
    full_path = os.path.join(PROJECT_ROOT, file_path)
    if os.path.exists(full_path):
        existing_files.append(file_path)
        print(f"  ✅ {file_path}")
    else:
        missing_files.append(file_path)
        print(f"  ❌ {file_path} - MISSING")

print(f"\n  Files: {len(existing_files)}/{len(REQUIRED_FILES)}")

if missing_files:
    print(f"\n  ⚠️  Missing {len(missing_files)} files!")
    print("  Please create these files before proceeding.")
else:
    print("  ✅ All files present!")

# ========== STEP 2: Check Imports ==========
print("\n📦 STEP 2: Checking Imports")
print("-" * 50)

import_errors = []

# Check config
try:
    from src.config.settings import (
        BASE_DIR, FAISS_INDEX_DIR, SERVER_HOST, SERVER_PORT,
        TWILIO_ACCOUNT_SID, DEEPGRAM_API_KEY, CARTESIA_API_KEY,
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    )
    print("  ✅ src.config.settings")
except Exception as e:
    import_errors.append(("src.config.settings", str(e)))
    print(f"  ❌ src.config.settings - {e}")

# Check utils
try:
    from src.utils.logger import get_logger
    logger = get_logger("test")
    print("  ✅ src.utils.logger")
except Exception as e:
    import_errors.append(("src.utils.logger", str(e)))
    print(f"  ❌ src.utils.logger - {e}")

# Check services
try:
    from src.services.faiss_service import faiss_service
    print("  ✅ src.services.faiss_service")
except Exception as e:
    import_errors.append(("src.services.faiss_service", str(e)))
    print(f"  ❌ src.services.faiss_service - {e}")

try:
    from src.services.llm_service import llm_service
    print("  ✅ src.services.llm_service")
except Exception as e:
    import_errors.append(("src.services.llm_service", str(e)))
    print(f"  ❌ src.services.llm_service - {e}")

try:
    from src.services.stt_service import stt_service
    print("  ✅ src.services.stt_service")
except Exception as e:
    import_errors.append(("src.services.stt_service", str(e)))
    print(f"  ❌ src.services.stt_service - {e}")

try:
    from src.services.tts_service import tts_service
    print("  ✅ src.services.tts_service")
except Exception as e:
    import_errors.append(("src.services.tts_service", str(e)))
    print(f"  ❌ src.services.tts_service - {e}")

# Check tools
try:
    from src.tools import TOOL_FUNCTIONS, TOOL_DEFINITIONS
    print(f"  ✅ src.tools - {len(TOOL_FUNCTIONS)} tools, {len(TOOL_DEFINITIONS)} definitions")
except Exception as e:
    import_errors.append(("src.tools", str(e)))
    print(f"  ❌ src.tools - {e}")

try:
    from src.tools.search_calls import search_calls, search_calls_tool
    print("  ✅ src.tools.search_calls")
except Exception as e:
    import_errors.append(("src.tools.search_calls", str(e)))
    print(f"  ❌ src.tools.search_calls - {e}")

try:
    from src.tools.get_call_details import get_call_details, get_all_calls, get_data_schema
    print("  ✅ src.tools.get_call_details")
except Exception as e:
    import_errors.append(("src.tools.get_call_details", str(e)))
    print(f"  ❌ src.tools.get_call_details - {e}")

try:
    from src.tools.filter_calls import filter_calls, get_unique_values
    print("  ✅ src.tools.filter_calls")
except Exception as e:
    import_errors.append(("src.tools.filter_calls", str(e)))
    print(f"  ❌ src.tools.filter_calls - {e}")

try:
    from src.tools.summarize import get_call_summary, get_dataset_overview, compare_calls
    print("  ✅ src.tools.summarize")
except Exception as e:
    import_errors.append(("src.tools.summarize", str(e)))
    print(f"  ❌ src.tools.summarize - {e}")

# Check agent
try:
    from src.agent.prompts import SYSTEM_PROMPT, VOICE_RESPONSE_PROMPT
    print("  ✅ src.agent.prompts")
except Exception as e:
    import_errors.append(("src.agent.prompts", str(e)))
    print(f"  ❌ src.agent.prompts - {e}")

try:
    from src.agent.state import AgentState, create_initial_state
    print("  ✅ src.agent.state")
except Exception as e:
    import_errors.append(("src.agent.state", str(e)))
    print(f"  ❌ src.agent.state - {e}")

try:
    from src.agent.graph import agent, run_agent, build_agent_graph
    print("  ✅ src.agent.graph")
except Exception as e:
    import_errors.append(("src.agent.graph", str(e)))
    print(f"  ❌ src.agent.graph - {e}")

try:
    from src.agent import run_agent
    print("  ✅ src.agent")
except Exception as e:
    import_errors.append(("src.agent", str(e)))
    print(f"  ❌ src.agent - {e}")

# Check voice
try:
    from src.voice.handlers import simple_voice_handler, voice_call_handler
    print("  ✅ src.voice.handlers")
except Exception as e:
    import_errors.append(("src.voice.handlers", str(e)))
    print(f"  ❌ src.voice.handlers - {e}")

try:
    from src.voice.pipeline import AgentProcessor
    print("  ✅ src.voice.pipeline")
except Exception as e:
    import_errors.append(("src.voice.pipeline", str(e)))
    print(f"  ❌ src.voice.pipeline - {e}")

try:
    from src.voice import simple_voice_handler
    print("  ✅ src.voice")
except Exception as e:
    import_errors.append(("src.voice", str(e)))
    print(f"  ❌ src.voice - {e}")

# Check main src
try:
    from src import run_agent, faiss_service, TOOL_FUNCTIONS
    print("  ✅ src (main)")
except Exception as e:
    import_errors.append(("src", str(e)))
    print(f"  ❌ src - {e}")

# ========== STEP 3: Check Environment Variables ==========
print("\n🔐 STEP 3: Checking Environment Variables")
print("-" * 50)

env_vars = [
    ("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID if 'TWILIO_ACCOUNT_SID' in dir() else None),
    ("TWILIO_AUTH_TOKEN", os.getenv("TWILIO_AUTH_TOKEN")),
    ("TWILIO_PHONE_NUMBER", os.getenv("TWILIO_PHONE_NUMBER")),
    ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY if 'DEEPGRAM_API_KEY' in dir() else None),
    ("CARTESIA_API_KEY", CARTESIA_API_KEY if 'CARTESIA_API_KEY' in dir() else None),
    ("CARTESIA_VOICE_ID", os.getenv("CARTESIA_VOICE_ID")),
    ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY if 'AZURE_OPENAI_API_KEY' in dir() else None),
    ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT if 'AZURE_OPENAI_ENDPOINT' in dir() else None),
    ("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT")),
    ("AZURE_OPENAI_EMBEDDING", os.getenv("AZURE_OPENAI_EMBEDDING")),
]

missing_env = []
for var_name, var_value in env_vars:
    if var_value:
        masked = var_value[:8] + "..." if len(str(var_value)) > 8 else var_value
        print(f"  ✅ {var_name}: {masked}")
    else:
        missing_env.append(var_name)
        print(f"  ❌ {var_name}: NOT SET")

# ========== STEP 4: Check FAISS Data ==========
print("\n📊 STEP 4: Checking FAISS Data")
print("-" * 50)

try:
    from src.services.faiss_service import faiss_service
    
    records = faiss_service.get_all_records()
    categories = faiss_service.get_categories()
    
    print(f"  ✅ Total records: {len(records)}")
    print(f"  ✅ Categories: {len(categories)}")
    print(f"  ✅ Fields per record: {len(records[0].keys()) if records else 0}")
    print(f"  ✅ Index vectors: {faiss_service.index.ntotal}")
except Exception as e:
    print(f"  ❌ FAISS check failed: {e}")

# ========== STEP 5: Check Tools ==========
print("\n🔧 STEP 5: Checking Tools")
print("-" * 50)

try:
    from src.tools import TOOL_FUNCTIONS, TOOL_DEFINITIONS
    
    print(f"  Tool Functions ({len(TOOL_FUNCTIONS)}):")
    for name in TOOL_FUNCTIONS.keys():
        print(f"    ✅ {name}")
    
    print(f"\n  Tool Definitions ({len(TOOL_DEFINITIONS)}):")
    for tool_def in TOOL_DEFINITIONS:
        name = tool_def.get("function", {}).get("name", "unknown")
        print(f"    ✅ {name}")
    
    # Verify they match
    func_names = set(TOOL_FUNCTIONS.keys())
    def_names = set(t.get("function", {}).get("name", "") for t in TOOL_DEFINITIONS)
    
    if func_names == def_names:
        print(f"\n  ✅ All tools properly matched!")
    else:
        print(f"\n  ⚠️  Mismatch detected!")
        print(f"      Functions only: {func_names - def_names}")
        print(f"      Definitions only: {def_names - func_names}")
        
except Exception as e:
    print(f"  ❌ Tools check failed: {e}")

# ========== STEP 6: Test Agent ==========
print("\n🤖 STEP 6: Testing Agent")
print("-" * 50)

try:
    from src.agent import run_agent
    
    test_response = run_agent("What tools do you have access to?", {})
    
    if test_response:
        print(f"  ✅ Agent responded: {test_response[:100]}...")
    else:
        print("  ⚠️  Agent returned empty response")
        
except Exception as e:
    print(f"  ❌ Agent test failed: {e}")

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("  VERIFICATION SUMMARY")
print("=" * 70)

total_issues = len(missing_files) + len(import_errors) + len(missing_env)

print(f"\n  📁 Files: {len(existing_files)}/{len(REQUIRED_FILES)}")
print(f"  📦 Import Errors: {len(import_errors)}")
print(f"  🔐 Missing Env Vars: {len(missing_env)}")

if total_issues == 0:
    print("\n  ✅ ALL CHECKS PASSED!")
    print("  ThubIQ Healthcare is ready to run.")
    print("\n  Start with:")
    print("    python bot.py --mode interactive")
    print("    python bot.py --mode server")
else:
    print(f"\n  ⚠️  {total_issues} ISSUES FOUND")
    
    if missing_files:
        print("\n  Missing Files:")
        for f in missing_files:
            print(f"    - {f}")
    
    if import_errors:
        print("\n  Import Errors:")
        for name, error in import_errors:
            print(f"    - {name}: {error}")
    
    if missing_env:
        print("\n  Missing Environment Variables:")
        for v in missing_env:
            print(f"    - {v}")

print("\n" + "=" * 70)