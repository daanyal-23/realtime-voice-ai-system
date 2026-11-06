"""
Unit tests for llm.py with mocked clients.
Tests token streaming, reasoning delta filtering, fallback logic, error states, and timeouts.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import LLM_FALLBACK_MODEL, LLM_MODEL
from llm import LLMError, _is_gpt_oss_model, stream_llm_response


def make_chunk(content=None, reasoning=None):
    """Helper to build a mock streaming chunk."""
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


class MockAsyncStream:
    """Async iterator simulating Groq API chunk stream."""

    def __init__(self, chunks, delay=0.0, raise_after=None):
        self.chunks = list(chunks)
        self.delay = delay
        self.raise_after = raise_after
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.raise_after is not None and self.idx >= self.raise_after:
            raise Exception(f"Simulated mid-stream failure after {self.idx} chunks")
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
        assert not any("Analyzing" in t or "Deciding" in t for t in tokens)

    @pytest.mark.asyncio
    async def test_primary_model_exception_triggers_fallback_when_no_content(self):
        """An exception in the primary model before any content triggers the fallback model."""
        mock_client = MagicMock()

        fallback_chunks = [
            make_chunk(content="Fallback"),
            make_chunk(content=" answer"),
        ]

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
    async def test_primary_fails_after_content_yielded_ends_stream_without_fallback(self):
        """If primary fails after content was already yielded, stream ends without fallback."""
        mock_client = MagicMock()

        # Stream yields one token, then raises an exception on chunk 2
        chunks = [
            make_chunk(content="Partial answer"),
            make_chunk(content=" never reached"),
        ]
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks, raise_after=1)
        )

        history = [{"role": "user", "content": "Continue answer"}]
        tokens = [
            t
            async for t in stream_llm_response(
                history, session_id="test", use_rag=False, client=mock_client
            )
        ]

        # Partial token received; stream gracefully closed without calling fallback
        assert tokens == ["Partial answer"]
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_idle_timeout_after_content_yielded_ends_stream_without_fallback(self):
        """If gap between chunks exceeds LLM_STREAM_IDLE_TIMEOUT_S after content was yielded, stream ends without fallback."""
        mock_client = MagicMock()

        chunks = [
            make_chunk(content="First sentence."),
            make_chunk(content="Second sentence delayed."),
        ]
        # Delay 0.3s between chunks
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks, delay=0.3)
        )

        history = [{"role": "user", "content": "Explain"}]
        # Set idle timeout to 0.05s so delay of 0.3s triggers idle timeout
        with patch("llm.LLM_STREAM_IDLE_TIMEOUT_S", 0.05):
            tokens = [
                t
                async for t in stream_llm_response(
                    history, session_id="test", use_rag=False, client=mock_client
                )
            ]

        # Yielded the first token, timed out waiting for the second, ended without fallback
        assert tokens == ["First sentence."]
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_healthy_long_stream_not_cut_by_initial_timeout(self):
        """A healthy long response is never cut by LLM_TIMEOUT_S as long as idle gaps stay low."""
        mock_client = MagicMock()

        # 4 chunks, each taking 0.02s -> total duration ~0.08s
        chunks = [
            make_chunk(content="Part 1 "),
            make_chunk(content="Part 2 "),
            make_chunk(content="Part 3 "),
            make_chunk(content="Part 4"),
        ]
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks, delay=0.02)
        )

        history = [{"role": "user", "content": "Give a detailed response"}]
        # Set initial TTFT timeout to 0.04s (first token arrives in 0.02s < 0.04s)
        # Total duration is ~0.08s > 0.04s, but idle timeout is 0.5s > 0.02s
        with patch("llm.LLM_TIMEOUT_S", 0.04), patch("llm.LLM_STREAM_IDLE_TIMEOUT_S", 0.5):
            tokens = [
                t
                async for t in stream_llm_response(
                    history, session_id="test", use_rag=False, client=mock_client
                )
            ]

        # All 4 tokens yielded successfully without being cut off by LLM_TIMEOUT_S
        assert tokens == ["Part 1 ", "Part 2 ", "Part 3 ", "Part 4"]
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_both_models_failing_produces_identifiable_error_state(self):
        """When both primary and fallback fail with no content, LLMError is raised."""
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
    async def test_timeout_before_first_content_triggers_fallback(self):
        """Timeout on primary before first content token triggers fallback to secondary model."""
        mock_client = MagicMock()

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


class TestConfig:
    def test_config_calls_load_dotenv(self):
        """Ensure config.py calls load_dotenv() upon import."""
        with patch("dotenv.load_dotenv") as mock_load_dotenv:
            importlib.reload(config)
            mock_load_dotenv.assert_called()
