"""
Tests for Speech-to-Text (STT) and Whisper Vocabulary Hint (Task S1).

Verifies acceptance criteria:
- Whisper prompt is passed correctly to Groq audio.transcriptions.create.
- Vocabulary hint includes fictional GP demo terms:
  - Oak Lane Medical Practice
  - Repeat prescription
  - Working days
  - Appointments
  - NHS 111
- Custom prompt override is respected when provided.
- Whisper transcript is extracted cleanly without modifying spoken words.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from stt import transcribe_audio, WHISPER_PROMPT, is_non_speech_caption


@pytest.mark.asyncio
class TestSTTWhisperPrompt:

    async def test_whisper_prompt_contains_demo_vocabulary(self):
        """WHISPER_PROMPT contains all required fictional GP demo terms."""
        required_terms = [
            "Oak Lane Medical Practice",
            "Repeat prescription",
            "Working days",
            "Appointments",
            "NHS 111",
        ]
        prompt_lower = WHISPER_PROMPT.lower()
        for term in required_terms:
            assert term.lower() in prompt_lower, f"Missing required term '{term}' in WHISPER_PROMPT"

    async def test_transcribe_audio_passes_prompt_to_groq_api(self):
        """transcribe_audio() passes the vocabulary prompt to active_client.audio.transcriptions.create."""
        mock_response = "How do I order a repeat prescription?"
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        pcm_data = b"\x01\x00" * 3200  # 100ms of audio

        with patch("stt.get_client", return_value=mock_client):
            result = await transcribe_audio(pcm_data, session_id="test_prompt_pass")

            assert result == mock_response
            assert mock_client.audio.transcriptions.create.call_count == 1

            _, kwargs = mock_client.audio.transcriptions.create.call_args
            assert "prompt" in kwargs
            assert kwargs["prompt"] == WHISPER_PROMPT
            assert kwargs["model"] == "whisper-large-v3-turbo"
            assert kwargs["language"] == "en"

    async def test_transcribe_audio_supports_custom_prompt_override(self):
        """Caller can provide a custom prompt override if needed."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value="I need an appointment.")
        custom_prompt = "Emergency triage chest pain 999."

        pcm_data = b"\x01\x00" * 3200

        with patch("stt.get_client", return_value=mock_client):
            result = await transcribe_audio(pcm_data, session_id="test_custom_prompt", prompt=custom_prompt)

            assert result == "I need an appointment."
            _, kwargs = mock_client.audio.transcriptions.create.call_args
            assert kwargs["prompt"] == custom_prompt


class TestNonSpeechCaptionFilter:
    """Verifies that standalone Whisper non-speech captions are filtered out while spoken sentences are kept."""

    @pytest.mark.parametrize(
        "caption",
        [
            "Music playing.",
            "Music playing",
            "[Music]",
            "[music]",
            "(music)",
            "(Music)",
            "[Applause]",
            "[laughter]",
            "(laughter)",
            "[Silence]",
            "[giggle]",
            "  Music playing.  \n",
            "[cheering]!",
        ],
    )
    def test_known_non_speech_captions_are_identified(self, caption):
        """Known non-speech captions evaluate to True for is_non_speech_caption."""
        assert is_non_speech_caption(caption) is True

    @pytest.mark.parametrize(
        "sentence",
        [
            "Can you turn off the music?",
            "I like classical music.",
            "There was applause after the speech.",
            "I heard laughter outside.",
            "Hello, I would like to book an appointment with a doctor.",
            "Can I order a repeat prescription please?",
            "Good morning Oak Lane Medical Practice.",
            "The music was too loud at the concert.",
            "Laughter is the best medicine.",
        ],
    )
    def test_legitimate_spoken_sentences_are_preserved(self, sentence):
        """Legitimate sentences mentioning music, applause, or laughter must NOT be filtered."""
        assert is_non_speech_caption(sentence) is False

    @pytest.mark.asyncio
    async def test_transcribe_audio_returns_empty_string_for_non_speech_caption(self):
        """transcribe_audio returns empty string when Whisper produces a non-speech caption."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value="Music playing.")

        pcm_data = b"\x01\x00" * 3200
        with patch("stt.get_client", return_value=mock_client):
            result = await transcribe_audio(pcm_data, session_id="test_filter_music")
            assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_audio_preserves_legitimate_speech_mentioning_music(self):
        """transcribe_audio preserves speech that mentions music."""
        mock_client = MagicMock()
        speech_text = "Can you turn off the music?"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=speech_text)

        pcm_data = b"\x01\x00" * 3200
        with patch("stt.get_client", return_value=mock_client):
            result = await transcribe_audio(pcm_data, session_id="test_keep_speech")
            assert result == speech_text


