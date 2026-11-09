"""
Configuration settings for Voice AI system.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Ensure .env values are loaded regardless of import order
load_dotenv()

# LLM Configuration
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_MAX_COMPLETION_TOKENS: int = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "400"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "4"))
LLM_STREAM_IDLE_TIMEOUT_S: float = float(os.getenv("LLM_STREAM_IDLE_TIMEOUT_S", "5"))

# VAD Configuration
VAD_FRAME_SIZE: int = int(os.getenv("VAD_FRAME_SIZE", "320"))  # 20ms at 16kHz
VAD_SPEECH_THRESHOLD: float = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.02"))
VAD_SILENCE_THRESHOLD: float = float(os.getenv("VAD_SILENCE_THRESHOLD", "0.01"))
VAD_SPEECH_CONFIRM_FRAMES: int = int(os.getenv("VAD_SPEECH_CONFIRM_FRAMES", "5"))   # 100ms
VAD_SILENCE_CONFIRM_FRAMES: int = int(os.getenv("VAD_SILENCE_CONFIRM_FRAMES", "15")) # 300ms
VAD_EMA_ALPHA: float = float(os.getenv("VAD_EMA_ALPHA", "0.05"))

# TTS Configuration
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "sapi5")
TTS_VOICE: str = os.getenv("TTS_VOICE", "Hazel")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "cedar")  # cedar or marin
OPENAI_TTS_SPEED: float = float(os.getenv("OPENAI_TTS_SPEED", "1.15"))
OPENAI_TTS_INSTRUCTIONS: str = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    "Speak in a warm, professional British English receptionist voice. Use a natural, confident, moderately brisk conversational pace. Avoid unnecessarily long pauses. Keep pronunciation clear and reassuring."
)
