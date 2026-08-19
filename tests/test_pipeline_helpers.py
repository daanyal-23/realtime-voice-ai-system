"""
Tests for helper functions in pipeline.py.
Fully deterministic pure function tests.
"""

import pytest
from pipeline import _is_sentence_boundary


class TestIsSentenceBoundary:
    @pytest.mark.parametrize("empty_input", [
        "",
        "   ",
        "\t\n",
        "  \n  ",
    ])
    def test_empty_or_whitespace_returns_false(self, empty_input: str):
        """Empty or whitespace-only strings return False."""
        assert _is_sentence_boundary(empty_input) is False

    @pytest.mark.parametrize("sentence", [
        "Hello world.",
        "This is a test.",
        "Sentence ending with period.   ",
        "Multiple. Sentences. In. One.",
    ])
    def test_period_boundary_returns_true(self, sentence: str):
        """Sentences ending in '.' return True."""
        assert _is_sentence_boundary(sentence) is True

    @pytest.mark.parametrize("sentence", [
        "What is your name?",
        "Are you ready?",
        "Really?  ",
    ])
    def test_question_mark_boundary_returns_true(self, sentence: str):
        """Sentences ending in '?' return True."""
        assert _is_sentence_boundary(sentence) is True

    @pytest.mark.parametrize("sentence", [
        "Great job!",
        "Watch out!",
        "Hello! ",
    ])
    def test_exclamation_mark_boundary_returns_true(self, sentence: str):
        """Sentences ending in '!' return True."""
        assert _is_sentence_boundary(sentence) is True

    def test_comma_short_text_returns_false(self):
        """Comma boundary returns False when text length is 40 characters or fewer."""
        # Short comma phrases
        assert _is_sentence_boundary("Hello,") is False
        assert _is_sentence_boundary("First,") is False
        assert _is_sentence_boundary("In addition,") is False

        # Exactly 40 characters ending with a comma: len("a" * 39 + ",") == 40
        text_40_chars = "a" * 39 + ","
        assert len(text_40_chars) == 40
        assert _is_sentence_boundary(text_40_chars) is False

    def test_comma_long_text_returns_true(self):
        """Comma boundary returns True when text length strictly exceeds 40 characters."""
        # Exactly 41 characters ending with a comma: len("a" * 40 + ",") == 41
        text_41_chars = "a" * 40 + ","
        assert len(text_41_chars) == 41
        assert _is_sentence_boundary(text_41_chars) is True

        # Realistic long clause ending with comma
        long_sentence_with_comma = "Although the weather was unexpectedly cold and rainy this morning,"
        assert len(long_sentence_with_comma) > 40
        assert _is_sentence_boundary(long_sentence_with_comma) is True

    @pytest.mark.parametrize("mid_sentence", [
        "Hello world",
        "This is an unfinished sentence without punctuation",
        "Hello. But continuing without ending punctuation",
        "Testing a string with, comma in the middle and no trailing punctuation",
        "A" * 100,  # long string but no punctuation at end
    ])
    def test_mid_sentence_without_ending_punctuation_returns_false(self, mid_sentence: str):
        """Mid-sentence text without sentence-ending punctuation returns False."""
        assert _is_sentence_boundary(mid_sentence) is False
