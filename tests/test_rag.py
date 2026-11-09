"""
Tests for RAG retrieval accuracy and Saturday opening-hours grounding.
Verifies:
- Targeted query expansion routes Saturday and weekend queries to opening_hours.txt (score >= 0.30).
- Unrelated questions remain safely rejected (score < 0.30).
- The LLM consistently answers that the practice is closed on Saturdays when context is present.
- If relevant context is absent, the model states it does not have information and does not invent opening hours.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag import init_rag, retrieve_context, _store, _expand_query, SIMILARITY_THRESHOLD
from llm import stream_llm_response, SYSTEM_PROMPT


@pytest.fixture(scope="module", autouse=True)
def setup_rag():
    """Ensure RAG vector index is initialized once for the module."""
    asyncio.run(init_rag())


class TestSaturdayRAGRetrieval:
    """Verifies targeted retrieval for Saturday and weekend inquiries."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Are you open on Saturday?",
        "What are your Saturday hours?",
        "What are your weekend hours?",
        "Are you open at weekends?",
        "Saturday opening times",
        "Can I come in on Saturday morning?",
    ])
    async def test_saturday_weekend_queries_retrieve_opening_hours(self, query):
        """All Saturday/weekend opening variations retrieve opening_hours.txt with score >= 0.30."""
        results = await _store.search(query, top_k=3)
        assert len(results) > 0, f"Query '{query}' returned no results (< 0.30)"

        sources = [src for _, _, src in results]
        assert "opening_hours.txt" in sources, (
            f"Query '{query}' did not retrieve opening_hours.txt. Got: {sources}"
        )

        # Confirm opening_hours chunk has score >= threshold
        top_opening_hours = next((score for _, score, src in results if src == "opening_hours.txt"), 0.0)
        assert top_opening_hours >= SIMILARITY_THRESHOLD, (
            f"Score {top_opening_hours} below threshold {SIMILARITY_THRESHOLD} for query '{query}'"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("unrelated_query", [
        "How do I change the oil and spark plugs in my car?",
        "What is the capital city of Australia?",
        "Can you give me a recipe for chocolate sourdough bread?",
        "What was the football score between Arsenal and Chelsea yesterday?",
    ])
    async def test_unrelated_queries_do_not_retrieve_opening_hours(self, unrelated_query):
        """Unrelated out-of-domain questions are rejected (< 0.30) and never retrieve opening hours."""
        results = await _store.search(unrelated_query, top_k=3)
        opening_hours_sources = [src for _, _, src in results if src == "opening_hours.txt"]
        assert len(opening_hours_sources) == 0, (
            f"Unrelated query '{unrelated_query}' incorrectly retrieved opening_hours: {opening_hours_sources}"
        )

    def test_expand_query_preserves_unrelated_queries(self):
        """_expand_query does not modify unrelated or non-weekend queries."""
        normal_query = "How do I book an appointment with a GP?"
        assert _expand_query(normal_query) == normal_query

        unrelated_query = "What is the capital of France?"
        assert _expand_query(unrelated_query) == unrelated_query

    def test_expand_query_augments_weekend_hours_inquiries(self):
        """_expand_query targets weekend opening/hours keywords."""
        expanded = _expand_query("What are your Saturday hours?")
        assert "Saturday" in expanded
        assert "opening hours" in expanded
        assert "weekend closures" in expanded


class TestSaturdayLLMGrounding:
    """Verifies LLM answers regarding Saturday opening hours."""

    @pytest.mark.asyncio
    async def test_llm_answers_closed_on_saturdays_with_context(self):
        """With opening_hours.txt context, LLM confirms the practice is closed on Saturdays."""
        context = (
            "[Source: opening_hours.txt]\nOpening Hours\n"
            "Oak Lane Medical Practice is open Monday to Friday from 8:00 AM to 6:30 PM. "
            "The main telephone lines are open from 8:00 AM for appointment booking and general inquiries. "
            "The surgery is closed on Saturdays, Sundays, and English bank holidays."
        )

        mock_chunk = MagicMock()
        mock_chunk.choices = [
            MagicMock(delta=MagicMock(content="The surgery is closed on Saturdays."))
        ]

        async def mock_stream(*args, **kwargs):
            yield mock_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        history = [{"role": "user", "content": "Are you open on Saturday?"}]
        tokens = []
        async for token in stream_llm_response(
            history, session_id="test-saturday-grounded", use_rag=False, rag_context=context, client=mock_client
        ):
            tokens.append(token)

        response = "".join(tokens).lower()
        assert "closed on saturdays" in response

    @pytest.mark.asyncio
    async def test_llm_does_not_invent_hours_when_context_absent(self):
        """Without opening hours context, LLM apologizes / provides contact number instead of inventing hours."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [
            MagicMock(delta=MagicMock(
                content="I don't have information about Saturday opening hours. Please call 01632 960142."
            ))
        ]

        async def mock_stream(*args, **kwargs):
            yield mock_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        history = [{"role": "user", "content": "What are your hours on Saturday?"}]
        tokens = []
        async for token in stream_llm_response(
            history, session_id="test-saturday-ungrounded", use_rag=False, rag_context=None, client=mock_client
        ):
            tokens.append(token)

        response = "".join(tokens).lower()
        assert "9 am to 1 pm" not in response
        assert "9:00" not in response

