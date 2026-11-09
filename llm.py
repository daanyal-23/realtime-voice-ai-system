"""
LLM streaming using Groq with fallback support and GPT-OSS reasoning handling.

Features:
  - Token streaming for low first-audio latency
  - Fallback from primary to secondary model on failure or empty completion
  - Handling of reasoning-only deltas for GPT-OSS models
  - Configurable timeouts (TTFT timeout and inter-chunk idle timeout)
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
    LLM_STREAM_IDLE_TIMEOUT_S,
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
        try:
            _default_client = AsyncGroq(max_retries=0)
        except TypeError:
            logger.warning("Groq SDK does not support max_retries parameter")
            _default_client = AsyncGroq()
    return _default_client


def __getattr__(name: str):
    """Provide lazy access to module-level client without instantiating at import time."""
    if name == "client":
        return get_default_client()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


async def _close_stream(stream: Optional[object]) -> None:
    """Close an async stream safely, supporting both aclose and close (sync or async)."""
    if stream is None:
        return
    try:
        if hasattr(stream, "aclose") and callable(stream.aclose):
            res = stream.aclose()
            if asyncio.iscoroutine(res):
                await res
        elif hasattr(stream, "close") and callable(stream.close):
            res = stream.close()
            if asyncio.iscoroutine(res):
                await res
    except Exception as e:
        logger.warning(f"Error closing stream: {e}")

SYSTEM_PROMPT = """You are an automated voice assistant for Oak Lane Medical Practice, speaking with a caller in British English.
Your role is to provide general practice information in short, clear, natural spoken sentences suitable for voice.
Speak plainly and warmly. Do not use markdown, bullet points, asterisks, or lists. Ask only one question at a time. If anything is unclear, ask the caller to repeat it.

Strict Grounding and Safety Rules:
1. Automated Identity: State that you are an automated assistant only at the beginning of the conversation or when directly asked about your identity or staff transfer. Do not prepend "I am an automated assistant" to every response. You are not a doctor or human receptionist.
2. Context Grounding: Answer strictly using facts from the provided practice context. If information is not in the context, explicitly state that you do not have that information and offer the practice telephone number (01632 960142) or suggest speaking with a member of staff. Never invent, assume, or speculate on practice opening hours, Saturday/weekend hours, or clinic schedules not explicitly provided in the context.
3. No Clinical Advice: Never provide medical diagnosis, clinical triage, or medication advice.
4. No Direct Actions: Never claim to book, cancel, or reschedule appointments, access medical records, or execute tasks yourself. Direct the caller on how they can do so (e.g. by calling 01632 960142 or using the NHS App).
5. Human Transfer Requests: If the caller asks to speak with a person or staff member, explain that you are an automated assistant and that while practice staff can help, this demo cannot transfer telephone calls. Provide the practice number 01632 960142.
6. Emergencies: If the caller describes a life-threatening medical emergency (such as severe chest pain, breathing difficulties, suspected stroke, heavy bleeding, or collapse), immediately advise them to hang up and dial 999."""


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
    rag_context: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens with automatic fallback and GPT-OSS reasoning support.

    Args:
        conversation_history: List of {role, content} dicts
        session_id: Identifier for session-scoped logging
        use_rag: Whether to inject RAG context
        client: Optional AsyncGroq client instance (for testing/dependency injection)
        rag_context: Optional pre-retrieved RAG context string

    Yields:
        Text tokens as they are generated

    Raises:
        LLMError: If both primary and fallback models fail or produce no content.
    """
    active_client = client or get_default_client()

    messages: List[Dict] = []

    # Build system prompt, optionally enhanced with RAG
    system_content = SYSTEM_PROMPT

    if rag_context:
        system_content += f"\n\nPractice Knowledge Base Context:\n{rag_context}"
        logger.info(f"[{session_id}] RAG context injected ({len(rag_context)} chars)")
    elif use_rag and conversation_history:
        last_user_message = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
            None
        )
        if last_user_message:
            try:
                retrieved_context = await retrieve_context(last_user_message)
                if retrieved_context:
                    system_content += f"\n\nPractice Knowledge Base Context:\n{retrieved_context}"
                    logger.info(f"[{session_id}] RAG context injected ({len(retrieved_context)} chars)")
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
        token_count = 0
        first_token_received = False
        first_chunk_content = None
        stream = None
        stream_iter = None
        terminal_finish_reason: Optional[str] = None

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
            # Phase 1: Bounded by LLM_TIMEOUT_S (Time-to-first-content-token)
            # No yield inside the timeout block: fetch chunk inside, exit block, then yield
            try:
                async with asyncio.timeout(LLM_TIMEOUT_S):
                    stream = await active_client.chat.completions.create(**create_kwargs)
                    stream_iter = stream.__aiter__()
                    while not first_token_received:
                        try:
                            chunk = await anext(stream_iter)
                        except StopAsyncIteration:
                            break

                        if not chunk.choices:
                            continue

                        choice = chunk.choices[0]
                        finish_reason = getattr(choice, "finish_reason", None)
                        if finish_reason is not None:
                            terminal_finish_reason = finish_reason

                        delta = choice.delta
                        content = getattr(delta, "content", None)
                        if content:
                            first_chunk_content = content
                            first_token_received = True
                            latency_ms = (time.perf_counter() - t_start) * 1000
                            logger.info(
                                f"[{session_id}] Isolated first-content-token latency for {model}: {latency_ms:.1f}ms"
                            )
                            token_count += 1
                            break

            except TimeoutError:
                logger.warning(
                    f"[{session_id}] Model {model} timed out waiting for first content token after {LLM_TIMEOUT_S}s"
                )
                # 0 tokens yielded -> continue to fallback (finally closes stream first)
                continue
            except (Exception, asyncio.CancelledError) as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                logger.warning(
                    f"[{session_id}] Model {model} failed before first content token: {type(e).__name__}: {e}"
                )
                # 0 tokens yielded -> continue to fallback (finally closes stream first)
                continue

            # If stream closed without yielding any content (e.g. empty or reasoning only)
            if not first_token_received:
                logger.warning(
                    f"[{session_id}] Model {model} produced 0 content tokens (finish_reason='{terminal_finish_reason}')"
                )
                # 0 tokens yielded -> continue to fallback (finally closes stream first)
                continue

            # Yield the first token OUTSIDE the asyncio.timeout block
            yield first_chunk_content

            # Phase 2: Stream remaining chunks bounded by LLM_STREAM_IDLE_TIMEOUT_S between chunks
            # Once content was already yielded, never start fallback (to prevent duplicate answer)
            if stream_iter is not None:
                while True:
                    try:
                        async with asyncio.timeout(LLM_STREAM_IDLE_TIMEOUT_S):
                            chunk = await anext(stream_iter)
                    except StopAsyncIteration:
                        # Stream completed
                        break
                    except TimeoutError:
                        logger.warning(
                            f"[{session_id}] Model {model} stream idle timeout after {LLM_STREAM_IDLE_TIMEOUT_S}s between chunks. Ending stream without fallback."
                        )
                        return
                    except (Exception, asyncio.CancelledError) as e:
                        if isinstance(e, asyncio.CancelledError):
                            raise
                        logger.warning(
                            f"[{session_id}] Model {model} failed mid-stream after yielding {token_count} tokens: {type(e).__name__}: {e}. Ending stream without fallback."
                        )
                        return

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason is not None:
                        terminal_finish_reason = finish_reason

                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    if content:
                        token_count += 1
                        yield content

                if terminal_finish_reason == "length":
                    logger.warning(
                        f"[{session_id}] Model {model} stream truncated by token limit: "
                        f"finish_reason='length' after {token_count} content tokens "
                        f"(max_completion_tokens={LLM_MAX_COMPLETION_TOKENS})."
                    )
                elif terminal_finish_reason is not None:
                    logger.info(
                        f"[{session_id}] Model {model} stream completed: "
                        f"finish_reason='{terminal_finish_reason}' ({token_count} content tokens)."
                    )
                else:
                    logger.info(
                        f"[{session_id}] Model {model} stream completed normally ({token_count} content tokens)."
                    )
                return

        finally:
            await _close_stream(stream)

    # If both primary and fallback models produced no content
    logger.error(
        f"[{session_id}] Both primary ({LLM_MODEL}) and fallback ({LLM_FALLBACK_MODEL}) models failed to generate content"
    )
    raise LLMError(
        f"Both primary model '{LLM_MODEL}' and fallback model '{LLM_FALLBACK_MODEL}' failed to generate content."
    )
