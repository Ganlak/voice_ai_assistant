import asyncio
from pipecat.transports.network.websocket_server import WebsocketServerTransport

from src.voice.pipeline import create_voice_pipeline, run_pipeline
from src.agent import run_agent
from src.services.tts_service import tts_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceCallHandler:
    """
    Handles voice call lifecycle.
    Agent manages conversation - handler manages connection.
    """
    
    def __init__(self):
        self.active_calls = {}
        self.context_store = {}
        logger.info("VoiceCallHandler initialized")
    
    async def handle_new_call(self, call_sid: str, transport: WebsocketServerTransport):
        """
        Handle new incoming call.
        Creates pipeline and starts processing.
        """
        logger.info(f"New call received: {call_sid}")
        
        try:
            # Initialize context for this call
            self.context_store[call_sid] = {
                "call_sid": call_sid,
                "turns": 0
            }
            
            # Create and run pipeline
            pipeline = await create_voice_pipeline(transport)
            self.active_calls[call_sid] = pipeline
            
            # Send greeting - Agent decides content
            greeting = await asyncio.to_thread(
                run_agent,
                "Generate a brief, friendly greeting for a healthcare call center.",
                {"is_greeting": True}
            )
            
            logger.info(f"Call {call_sid} pipeline started with greeting")
            
            # Run pipeline
            await run_pipeline(pipeline)
            
        except Exception as e:
            logger.error(f"Call handling failed for {call_sid}: {e}")
            await self.handle_call_error(call_sid, e)
        
        finally:
            await self.handle_call_end(call_sid)
    
    async def handle_call_end(self, call_sid: str):
        """
        Clean up when call ends.
        """
        logger.info(f"Call ended: {call_sid}")
        
        # Remove from active calls
        if call_sid in self.active_calls:
            del self.active_calls[call_sid]
        
        # Clear context
        if call_sid in self.context_store:
            del self.context_store[call_sid]
    
    async def handle_call_error(self, call_sid: str, error: Exception):
        """
        Handle call errors gracefully.
        """
        logger.error(f"Call error for {call_sid}: {error}")
        
        # Generate error message via Agent
        try:
            error_response = await asyncio.to_thread(
                run_agent,
                f"Generate a brief, polite message explaining a technical issue occurred. Error: {str(error)[:100]}",
                {"is_error": True}
            )
            logger.info(f"Error response generated for {call_sid}")
        except Exception:
            logger.error(f"Failed to generate error response for {call_sid}")
    
    def get_call_context(self, call_sid: str) -> dict:
        """
        Get context for a specific call.
        """
        return self.context_store.get(call_sid, {})
    
    def update_call_context(self, call_sid: str, updates: dict):
        """
        Update context for a specific call.
        """
        if call_sid in self.context_store:
            self.context_store[call_sid].update(updates)
    
    def get_active_calls_count(self) -> int:
        """
        Get number of active calls.
        """
        return len(self.active_calls)


# Non-streaming handler for testing
class SimpleVoiceHandler:
    """
    Simple handler for testing without Twilio.
    Processes text input → Agent → Audio output.
    """
    
    def __init__(self):
        self.context = {}
        logger.info("SimpleVoiceHandler initialized")
    
    async def process_text(self, text: str) -> bytes:
        """
        Process text input and return audio response.
        """
        logger.info(f"Processing text: {text}")
        
        try:
            # Run Agent
            response = await asyncio.to_thread(
                run_agent,
                text,
                self.context
            )
            
            # Update context
            self.context["last_query"] = text
            self.context["last_response"] = response
            
            # Generate audio
            audio = tts_service.synthesize(response)
            
            logger.info(f"Generated audio response: {len(audio)} bytes")
            
            return audio
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            raise
    
    def process_text_sync(self, text: str) -> str:
        """
        Synchronous text-only processing.
        Returns text response (no audio).
        """
        logger.info(f"Processing text (sync): {text}")
        
        response = run_agent(text, self.context)
        
        self.context["last_query"] = text
        self.context["last_response"] = response
        
        return response
    
    def reset_context(self):
        """
        Reset conversation context.
        """
        self.context = {}
        logger.info("Context reset")


# Singleton instances
voice_call_handler = VoiceCallHandler()
simple_voice_handler = SimpleVoiceHandler()