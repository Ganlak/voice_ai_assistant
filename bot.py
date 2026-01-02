"""
PSR Voice AI Assistant
WellStreet Urgent Care

Main entry point for the voice bot application.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import SERVER_HOST, SERVER_PORT, TWILIO_PHONE_NUMBER
from src.agent.graph import run_agent, run_agent_with_history, get_greeting
from src.voice.handlers import simple_voice_handler
from src.utils.logger import get_logger

logger = get_logger(__name__)

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          ██████╗ ███████╗██████╗                              ║
║          ██╔══██╗██╔════╝██╔══██╗                             ║
║          ██████╔╝███████╗██████╔╝                             ║
║          ██╔═══╝ ╚════██║██╔══██╗                             ║
║          ██║     ███████║██║  ██║                             ║
║          ╚═╝     ╚══════╝╚═╝  ╚═╝                             ║
║                                                               ║
║               PSR Voice AI Assistant                          ║
║             WellStreet Urgent Care                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print application banner."""
    print(BANNER)
    print(f"  📞 Twilio Phone: {TWILIO_PHONE_NUMBER or 'Not configured'}")
    print(f"  🌐 Server: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  🧪 Test UI: http://localhost:{SERVER_PORT}/demo")
    print()


def run_interactive_mode():
    """
    Run in interactive CLI mode for testing.
    Type queries and get responses without voice.
    """
    print_banner()
    print("=" * 60)
    print("  INTERACTIVE MODE - Type 'quit' to exit")
    print("=" * 60)
    print()
    
    # Initial greeting
    greeting = get_greeting()
    print(f"🤖 Sarah: {greeting}\n")
    
    while True:
        try:
            user_input = input("👤 Caller: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                farewell = "Thank you for calling WellStreet Urgent Care. Have a great day!"
                print(f"\n🤖 Sarah: {farewell}")
                print("\nCall ended.\n")
                break
            
            # Run agent
            response = run_agent(user_input)
            print(f"\n🤖 Sarah: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nCall ended.\n")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {e}\n")


def run_server_mode():
    """
    Run in server mode for Twilio integration.
    """
    print_banner()
    print("=" * 60)
    print("  SERVER MODE - Starting FastAPI server")
    print("=" * 60)
    print()
    
    from server import start_server
    start_server()


def run_test_agent():
    """
    Quick test of the agent functionality.
    """
    print_banner()
    print("=" * 60)
    print("  TESTING PSR VOICE AI AGENT")
    print("=" * 60)
    print()
    
    test_queries = [
        "Hello, I need to schedule an appointment.",
        "Do you accept walk-ins?",
        "What's your policy for late arrivals?",
        "How do I get to your clinic?",
        "What are your wait times like?",
        "Can I reschedule my appointment?",
        "Thank you, goodbye!"
    ]
    
    for query in test_queries:
        print(f"👤 Caller: {query}")
        response = run_agent(query)
        print(f"🤖 Sarah: {response}")
        print("-" * 50)
        print()


def run_conversation_test():
    """
    Test multi-turn conversation with history.
    """
    print_banner()
    print("=" * 60)
    print("  TESTING MULTI-TURN CONVERSATION")
    print("=" * 60)
    print()
    
    conversation = [
        "Hi, I'd like to schedule an appointment.",
        "Actually, can I just walk in instead?",
        "What if I'm running late?",
        "Okay, thank you for your help!"
    ]
    
    history = []
    
    # Initial greeting
    print(f"🤖 Sarah: {get_greeting()}\n")
    
    for query in conversation:
        print(f"👤 Caller: {query}")
        response, history = run_agent_with_history(query, history)
        print(f"🤖 Sarah: {response}")
        print(f"   [Conversation history: {len(history)} messages]")
        print("-" * 50)
        print()


def main():
    """
    Main entry point.
    Supports multiple modes: interactive, server, test, conversation
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PSR Voice AI Assistant - WellStreet Urgent Care"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "server", "test", "conversation"],
        default="interactive",
        help="Run mode: interactive (CLI), server (FastAPI), test (quick test), or conversation (multi-turn test)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "interactive":
            run_interactive_mode()
        elif args.mode == "server":
            run_server_mode()
        elif args.mode == "test":
            run_test_agent()
        elif args.mode == "conversation":
            run_conversation_test()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()