"""
LLM streaming using Groq (llama-3.3-70b-versatile).

Why Groq:
  - Free tier — generous daily limits
  - llama-3.3-70b-versatile: GPT-4o quality, ~900 tokens/sec on Groq hardware
  - True token streaming (same interface as OpenAI)
  - No local GPU needed for LLM

Features:
  - Token streaming for low first-audio latency
  - System prompt tuned for voice (no markdown, concise)
  - Optional RAG context injection
  - Conversation history for multi-turn dialogue
"""

import os
import logging
from typing import AsyncGenerator, List, Dict

from groq import AsyncGroq

from rag import retrieve_context
from dotenv import load_dotenv

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger("llm")
client = AsyncGroq()  # reads GROQ_API_KEY from environment

SYSTEM_PROMPT = """You are a helpful, conversational AI voice assistant.
Keep your responses concise and natural for spoken conversation.
Avoid using markdown, bullet points, or formatting — speak in plain sentences.
Be warm, direct, and helpful. If you don't know something, say so clearly."""


async def stream_llm_response(
    conversation_history: List[Dict],
    session_id: str = "",
    use_rag: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens via Groq.

    Args:
        conversation_history: List of {role, content} dicts
        session_id: For logging
        use_rag: Whether to inject RAG context

    Yields:
        Text tokens as they are generated
    """
    messages = []

    # Build system prompt, optionally enhanced with RAG
    system_content = SYSTEM_PROMPT

    if use_rag and conversation_history:
        last_user_message = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
            None
        )
        if last_user_message:
            rag_context = await retrieve_context(last_user_message)
            if rag_context:
                system_content += f"\n\nRelevant context from knowledge base:\n{rag_context}"
                logger.info(f"[{session_id}] RAG context injected ({len(rag_context)} chars)")

    messages.append({"role": "system", "content": system_content})
    messages.extend(conversation_history[-10:])

    try:
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=150,  # keep responses concise for voice
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except Exception as e:
        logger.error(f"[{session_id}] Groq LLM streaming error: {e}")
        raise