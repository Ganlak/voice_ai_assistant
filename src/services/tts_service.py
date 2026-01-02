"""
ElevenLabs TTS Service - Human-Like Voice
==========================================
"""

import os
from elevenlabs import ElevenLabs
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TTSService:
    """ElevenLabs Text-to-Speech Service."""
    
    def __init__(self):
        """Initialize ElevenLabs TTS."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        
        if not self.voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID not found in environment")
        
        self.client = ElevenLabs(api_key=self.api_key)
        
        logger.info(f"ElevenLabs TTS Service initialized with voice: {self.voice_id}")
    
    def get_voice_id(self):
        """Return voice ID."""
        return self.voice_id
    
    def get_api_key(self):
        """Return API key."""
        return self.api_key
    
    def synthesize(self, text: str) -> bytes:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes (MP3 format)
        """
        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )
            
            # Collect bytes from generator
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)
            
            audio_data = b''.join(audio_chunks)
            logger.debug(f"Synthesized {len(audio_data)} bytes for: {text[:50]}...")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed: {e}")
            raise


# Singleton instance
tts_service = TTSService()