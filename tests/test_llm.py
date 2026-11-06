"""
Unit tests for llm.py with mocked clients.
Tests token streaming, reasoning delta filtering, fallback logic, error states, and timeouts.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import LLM_FALLBACK_MODEL, LLM_MODEL
from llm import LLMError, _is_gpt_oss_model, stream_llm_response


def make_chunk(content=None, reasoning=None):
    """Helper to build a mock streaming chunk."""
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


class MockAsyncStream:
    """Async iterator simulating Groq API chunk stream."""

    def __init__(self, chunks, delay=0.0):
        self.chunks = list(chunks)
        self.delay = delay
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.idx >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.idx]
        self.idx += 1
        return chunk


class TestLLMStreaming:
    @pytest.mark.asyncio
    async def test_content_tokens_are_yielded(self):
        """Content tokens are properly yielded from the primary model stream."""
        mock_client = MagicMock()
        chunks = [
            make_chunk(content="Hello"),
            make_chunk(content=" world"),
            make_chunk(content="!"),
        ]
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks)
        )

        history = [{"role": "user", "content": "Hi"}]
        tokens = [
            t
            async for t in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            )
        ]

        assert tokens == ["Hello", " world", "!"]
        assert mock_client.chat.completions.create.call_count == 1
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == LLM_MODEL
        assert "max_completion_tokens" in call_kwargs

    @pytest.mark.asyncio
    async def test_reasoning_only_deltas_are_ignored(self):
        """Reasoning-only deltas (content=None or '') must not be yielded."""
        mock_client = MagicMock()
        chunks = [
            make_chunk(content="", reasoning=None),
            make_chunk(content=None, reasoning="Analyzing user query..."),
            make_chunk(content=None, reasoning="Deciding tone..."),
            make_chunk(content="Welcome", reasoning=None),
            make_chunk(content=" to the clinic.", reasoning=None),
        ]
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks)
        )

        history = [{"role": "user", "content": "I need help"}]
        tokens = [
            t
            async for t in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            )
        ]

        assert tokens == ["Welcome", " to the clinic."]
        # Ensure reasoning text was never leaked into yielded tokens
        assert not any("Analyzing" in t or "Deciding" in t for t in tokens)

    @pytest.mark.asyncio
    async def test_primary_model_exception_triggers_fallback(self):
        """An exception in the primary model triggers the configured fallback model."""
        mock_client = MagicMock()

        fallback_chunks = [
            make_chunk(content="Fallback"),
            make_chunk(content=" answer"),
        ]

        # Primary raises error, fallback succeeds
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Model temporarily overloaded 503"),
                MockAsyncStream(fallback_chunks),
            ]
        )

        history = [{"role": "user", "content": "Help"}]
        tokens = [
            t
            async for t in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            )
        ]

        assert tokens == ["Fallback", " answer"]
        assert mock_client.chat.completions.create.call_count == 2
        first_call = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert first_call["model"] == LLM_MODEL
        assert second_call["model"] == LLM_FALLBACK_MODEL

    @pytest.mark.asyncio
    async def test_empty_primary_stream_triggers_fallback(self):
        """When primary model produces 0 content tokens, fallback model is invoked."""
        mock_client = MagicMock()

        # Primary yields only reasoning / empty content
        empty_chunks = [
            make_chunk(content=None, reasoning="Internal thoughts only..."),
        ]
        fallback_chunks = [
            make_chunk(content="Recovered"),
            make_chunk(content=" content"),
        ]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                MockAsyncStream(empty_chunks),
                MockAsyncStream(fallback_chunks),
            ]
        )

        history = [{"role": "user", "content": "Status"}]
        tokens = [
            t
            async for t in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            )
        ]

        assert tokens == ["Recovered", " content"]
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_client.chat.completions.create.call_args_list[1].kwargs["model"] == LLM_FALLBACK_MODEL

    @pytest.mark.asyncio
    async def test_both_models_failing_produces_identifiable_error_state(self):
        """When both primary and fallback fail, LLMError is raised (no fake apologies)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Primary failed"),
                Exception("Fallback also failed"),
            ]
        )

        history = [{"role": "user", "content": "Book appointment"}]
        with pytest.raises(LLMError) as exc_info:
            async for _ in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            ):
                pass

        assert "failed to generate content" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_is_handled_correctly(self):
        """Timeout on primary triggers fallback to secondary model."""
        mock_client = MagicMock()

        # Primary hangs (delay 0.5s with timeout 0.05s)
        hanging_stream = MockAsyncStream(
            [make_chunk(content="Late")], delay=0.5
        )
        fallback_stream = MockAsyncStream(
            [make_chunk(content="On"), make_chunk(content=" time")]
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                hanging_stream,
                fallback_stream,
            ]
        )

        history = [{"role": "user", "content": "Ping"}]
        with patch("llm.LLM_TIMEOUT_S", 0.05):
            tokens = [
                t
                async for t in stream_llm_response(
                    history, session_id="test", use_rag=False, client=mock_client
                )
            ]

        assert tokens == ["On", " time"]
        assert mock_client.chat.completions.create.call_count == 2


class TestGPTOSSModelHelper:
    def test_gpt_oss_model_detection(self):
        """Check reasoning_effort filter for GPT-OSS model names."""
        assert _is_gpt_oss_model("openai/gpt-oss-120b") is True
        assert _is_gpt_oss_model("openai/gpt-oss-20b") is True
        assert _is_gpt_oss_model("GPT-OSS-test") is True
        assert _is_gpt_oss_model("llama-3.3-70b-versatile") is False
        assert _is_gpt_oss_model("llama-3.1-8b-instant") is False
