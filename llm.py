"""
LLM streaming using Groq with fallback support and GPT-OSS reasoning handling.

Features:
  - Token streaming for low first-audio latency
  - Fallback from primary to secondary model on failure or empty completion
  - Handling of reasoning-only deltas for GPT-OSS models
  - Configurable timeouts and completion token limits
  - Isolated first-content-token latency measurement
  - System prompt tuned for voice (no markdown, concise)
  - Optional RAG context injection
  - Conversation history for multi-turn dialogue
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from groq import AsyncGroq

from config import (
    LLM_FALLBACK_MODEL,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
)
from rag import retrieve_context

load_dotenv()

logger = logging.getLogger("llm")

# Default global client instance
_default_client: Optional[AsyncGroq] = None


def get_default_client() -> AsyncGroq:
    """Lazily initialize and return default AsyncGroq client."""
    global _default_client
    if _default_client is None:
        _default_client = AsyncGroq()
    return _default_client


# Preserved for backwards compatibility with any existing imports
client = get_default_client()

SYSTEM_PROMPT = """You are a helpful, conversational AI voice assistant.
Keep your responses concise and natural for spoken conversation.
Avoid using markdown, bullet points, or formatting — speak in plain sentences.
Be warm, direct, and helpful. If you don't know something, say so clearly."""


class LLMError(RuntimeError):
    """Raised when both primary and fallback LLM models fail or produce no content."""
    pass


def _is_gpt_oss_model(model_name: str) -> bool:
    """Check if model requires or supports reasoning_effort parameter."""
    return "gpt-oss" in model_name.lower()


async def stream_llm_response(
    conversation_history: List[Dict],
    session_id: str = "",
    use_rag: bool = True,
    client: Optional[AsyncGroq] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens with automatic fallback and GPT-OSS reasoning support.

    Args:
        conversation_history: List of {role, content} dicts
        session_id: Identifier for session-scoped logging
        use_rag: Whether to inject RAG context
        client: Optional AsyncGroq client instance (for testing/dependency injection)

    Yields:
        Text tokens as they are generated

    Raises:
        LLMError: If both primary and fallback models fail or produce no content.
    """
    active_client = client or get_default_client()

    messages: List[Dict] = []

    # Build system prompt, optionally enhanced with RAG
    system_content = SYSTEM_PROMPT

    if use_rag and conversation_history:
        last_user_message = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
            None
        )
        if last_user_message:
            try:
                rag_context = await retrieve_context(last_user_message)
                if rag_context:
                    system_content += f"\n\nRelevant context from knowledge base:\n{rag_context}"
                    logger.info(f"[{session_id}] RAG context injected ({len(rag_context)} chars)")
            except Exception as e:
                logger.warning(f"[{session_id}] RAG retrieval failed: {e}")

    messages.append({"role": "system", "content": system_content})
    messages.extend(conversation_history[-10:])

    models_to_try = [LLM_MODEL, LLM_FALLBACK_MODEL]

    for idx, model in enumerate(models_to_try):
        is_fallback = idx > 0
        if is_fallback:
            logger.warning(f"[{session_id}] Retrying with fallback model: {model}")
        else:
            logger.info(f"[{session_id}] Selected LLM model: {model}")

        t_start = time.perf_counter()
        first_token_logged = False
        token_count = 0

        create_kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": LLM_TEMPERATURE,
            "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS,
        }

        if _is_gpt_oss_model(model):
            create_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

        try:
            async with asyncio.timeout(LLM_TIMEOUT_S):
                stream = await active_client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        if not first_token_logged:
                            latency_ms = (time.perf_counter() - t_start) * 1000
                            logger.info(
                                f"[{session_id}] Isolated first-content-token latency for {model}: {latency_ms:.1f}ms"
                            )
                            first_token_logged = True
                        token_count += 1
                        yield content

            if token_count > 0:
                # Successfully streamed content tokens
                return
            else:
                logger.warning(f"[{session_id}] Model {model} produced 0 content tokens")

        except TimeoutError:
            logger.warning(f"[{session_id}] Model {model} timed out after {LLM_TIMEOUT_S}s")
            if token_count > 0:
                raise LLMError(f"Model {model} timed out mid-stream after yielding {token_count} tokens")
        except Exception as e:
            logger.warning(f"[{session_id}] Model {model} failed: {type(e).__name__}: {e}")
            if token_count > 0:
                raise LLMError(f"Model {model} failed mid-stream after yielding {token_count} tokens: {e}") from e

    # Both primary and fallback models failed
    logger.error(
        f"[{session_id}] Both primary ({LLM_MODEL}) and fallback ({LLM_FALLBACK_MODEL}) models failed to generate content"
    )
    raise LLMError(
        f"Both primary model '{LLM_MODEL}' and fallback model '{LLM_FALLBACK_MODEL}' failed to generate content."
    )
