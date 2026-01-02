from openai import AzureOpenAI
from src.config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.model = AZURE_OPENAI_DEPLOYMENT
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate response from LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_with_tools(self, messages: list, tools: list, temperature: float = 0.7) -> dict:
        """Generate response with tool calling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature
            )
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM tool generation failed: {e}")
            raise

# Singleton instance
llm_service = LLMService()