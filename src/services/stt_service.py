from deepgram import DeepgramClient
from src.config.settings import DEEPGRAM_API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)

class STTService:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        logger.info("Deepgram STT Service initialized")
    
    def get_client(self):
        """Return Deepgram client for Pipecat integration"""
        return self.client
    
    def get_api_key(self):
        """Return API key for Pipecat"""
        return DEEPGRAM_API_KEY

# Singleton instance
stt_service = STTService()