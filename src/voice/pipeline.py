import asyncio
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    EndFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.websocket.server import WebsocketServerTransport

from src.agent import run_agent
from src.config.settings import (
    DEEPGRAM_API_KEY,
    CARTESIA_API_KEY,
    CARTESIA_VOICE_ID
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentProcessor(FrameProcessor):
    """
    Custom processor that connects voice input to Agentic AI.
    Receives transcription → Runs Agent → Returns response for TTS.
    """
    
    def __init__(self):
        super().__init__()
        self.context = {}
        logger.info("AgentProcessor initialized")
    
    async def process_frame(self, frame: Frame, direction):
        """
        Process incoming frames.
        Agent handles all reasoning - no hardcoded logic here.
        """
        await super().process_frame(frame, direction)
        
        # Handle transcription from STT
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            
            if not user_text:
                return
            
            logger.info(f"Received transcription: {user_text}")
            
            try:
                # Run Agentic AI
                response = await asyncio.to_thread(
                    run_agent,
                    user_text,
                    self.context
                )
                
                # Update context for multi-turn
                self.context["last_query"] = user_text
                self.context["last_response"] = response
                
                # Send response to TTS
                await self.push_frame(TextFrame(text=response))
                
                logger.info(f"Agent response sent to TTS: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"Agent processing failed: {e}")
                error_response = "I'm sorry, I had trouble processing that. Could you please try again?"
                await self.push_frame(TextFrame(text=error_response))
        
        # Pass through other frames
        else:
            await self.push_frame(frame, direction)


async def create_voice_pipeline(transport: WebsocketServerTransport) -> Pipeline:
    """
    Create the voice pipeline with Pipecat.
    Connects: Transport → STT → Agent → TTS → Transport
    """
    logger.info("Creating voice pipeline")
    
    # Speech-to-Text (Deepgram)
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2",
        language="en",
        encoding="linear16",
        sample_rate=16000
    )
    
    # Agentic AI Processor
    agent_processor = AgentProcessor()
    
    # Text-to-Speech (Cartesia)
    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        voice_id=CARTESIA_VOICE_ID,
        model_id="sonic-english",
        sample_rate=24000
    )
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),   # Audio from Twilio
        stt,                 # Speech → Text
        agent_processor,     # Text → Agent → Response
        tts,                 # Response → Speech
        transport.output()   # Audio to Twilio
    ])
    
    logger.info("Voice pipeline created successfully")
    
    return pipeline


async def run_pipeline(pipeline: Pipeline):
    """
    Run the pipeline task.
    """
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True
        )
    )
    
    logger.info("Starting pipeline task")
    await task.run()