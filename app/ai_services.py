"""
AI Services Integration Module

This module provides integration with external AI services to avoid heavy local dependencies.
Uses the AI infrastructure services available in the Kubernetes cluster.
"""

import os
import httpx
import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AIServiceClient:
    """Client for external AI services"""

    def __init__(self):
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://192.168.0.20:8000")
        self.stt_service_url = os.getenv("STT_SERVICE_URL", "http://192.168.0.20:8002")
        self.tts_service_url = os.getenv("TTS_SERVICE_URL", "http://192.168.0.20:8003")
        self.routing_api_url = os.getenv("AI_ROUTING_API_URL", "http://192.168.0.20:8001")

        # HTTP client with timeout
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text using the vLLM service

        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        try:
            payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stream": False}

            response = await self.client.post(f"{self.vllm_base_url}/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("text", "")

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

    async def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> str:
        """
        Transcribe audio using the STT service

        Args:
            audio_data: Audio file data
            audio_format: Audio format (wav, mp3, etc.)

        Returns:
            Transcribed text
        """
        try:
            files = {"audio": ("audio." + audio_format, audio_data, f"audio/{audio_format}")}

            response = await self.client.post(f"{self.stt_service_url}/transcribe", files=files)
            response.raise_for_status()

            result = response.json()
            return result.get("text", "")

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error transcribing audio: {str(e)}"

    async def generate_speech(self, text: str, voice: str = "default") -> bytes:
        """
        Generate speech from text using the TTS service

        Args:
            text: Text to convert to speech
            voice: Voice to use for synthesis

        Returns:
            Audio data
        """
        try:
            payload = {"text": text, "voice": voice}

            response = await self.client.post(f"{self.tts_service_url}/synthesize", json=payload)
            response.raise_for_status()

            return response.content

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return b""

    async def get_embeddings(self, texts: List[str], model: str = "default") -> List[List[float]]:
        """
        Get embeddings for texts using the routing API

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        try:
            payload = {"texts": texts, "model": model}

            response = await self.client.post(f"{self.routing_api_url}/embeddings", json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("embeddings", [])

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all AI services

        Returns:
            Dictionary with service health status
        """
        health_status = {}

        services = {
            "vllm": self.vllm_base_url,
            "stt": self.stt_service_url,
            "tts": self.tts_service_url,
            "routing": self.routing_api_url,
        }

        for service_name, service_url in services.items():
            try:
                response = await self.client.get(f"{service_url}/health", timeout=5.0)
                health_status[service_name] = response.status_code == 200
            except Exception:
                health_status[service_name] = False

        return health_status


# Global AI service client instance
ai_client = AIServiceClient()


async def get_ai_client() -> AIServiceClient:
    """Get the global AI service client"""
    return ai_client


async def cleanup_ai_client():
    """Cleanup the AI service client"""
    await ai_client.close()
