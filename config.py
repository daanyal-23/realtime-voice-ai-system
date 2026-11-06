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
LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "15"))
LLM_STREAM_IDLE_TIMEOUT_S: float = float(os.getenv("LLM_STREAM_IDLE_TIMEOUT_S", "8"))
