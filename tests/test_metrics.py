"""
Unit tests for Honest Latency metrics (Task B4).

Verifies acceptance criteria:
1. Every required metric field is present in turn_end:
   - stt_ms
   - rag_ms
   - llm_ttft_ms
   - speech_end_to_first_audio_ms
   - total_turn_ms
2. llm_ttft_ms strictly measures from LLM request start to first token,
   excluding STT and RAG time.
3. speech_end_to_first_audio_ms accurately reflects time from user speech end
   until the first audio chunk is dispatched.
4. Empty transcripts and error turns emit well-formed metrics.
"""

import asyncio
import json
import time
from unittest.mock import patch

import pytest

from pipeline import VoicePipeline, MSG_TURN_START, MSG_TURN_END, MSG_TRANSCRIPT


class MockWebSocket:
    """Mock WebSocket capturing JSON control messages and binary audio chunks."""

    def __init__(self):
        self.sent_messages = []
        self.sent_binary = []

    async def send(self, data):
        if isinstance(data, (bytes, bytearray)):
            self.sent_binary.append(bytes(data))
        else:
            try:
                self.sent_messages.append(json.loads(data))
            except Exception:
                self.sent_messages.append(data)


@pytest.mark.asyncio
class TestPipelineMetrics:

    async def test_turn_metrics_payload_contains_all_fields(self):
        """Mocked turn asserts all 5 required latency fields are present and populated."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_metrics_all_fields", websocket=ws)

        async def mock_transcribe(audio_data, session_id=""):
            await asyncio.sleep(0.01)
            return "What are your opening hours?"

        async def mock_rag(query):
            await asyncio.sleep(0.01)
            return "We are open Monday to Friday 8:00 to 18:30."

        async def mock_llm(history, session_id="", rag_context=None):
            await asyncio.sleep(0.01)
            yield "We "
            yield "are open "
            yield "from 8am to 6:30pm."

        async def mock_tts(sentence, session_id=""):
            await asyncio.sleep(0.01)
            yield b"RIFF_WAV_HEADER_AND_PCM_DATA"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.retrieve_context", side_effect=mock_rag), \
             patch("pipeline.stream_llm_response", side_effect=mock_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            # Check messages captured
            msg_types = [m.get("type") for m in ws.sent_messages]
            assert MSG_TRANSCRIPT in msg_types
            assert "latency" in msg_types
            assert MSG_TURN_START in msg_types
            assert MSG_TURN_END in msg_types

            # Verify latency interim event
            latency_msg = next(m for m in ws.sent_messages if m.get("type") == "latency")
            assert "stt_ms" in latency_msg
            assert "rag_ms" in latency_msg
            assert "llm_ttft_ms" in latency_msg
            assert latency_msg["turn_id"] == 1

            # Verify turn_start event
            turn_start_msg = next(m for m in ws.sent_messages if m.get("type") == MSG_TURN_START)
            assert "speech_end_to_first_audio_ms" in turn_start_msg
            assert turn_start_msg["speech_end_to_first_audio_ms"] > 0.0

            # Verify turn_end event has all 5 fields
            turn_end_msg = next(m for m in ws.sent_messages if m.get("type") == MSG_TURN_END)
            required_fields = [
                "stt_ms",
                "rag_ms",
                "llm_ttft_ms",
                "speech_end_to_first_audio_ms",
                "total_turn_ms",
            ]
            for field in required_fields:
                assert field in turn_end_msg, f"Missing required field {field} in turn_end"
                assert isinstance(turn_end_msg[field], (int, float))
                assert turn_end_msg[field] > 0.0, f"Field {field} must be > 0.0"

            # Check backward compatibility field
            assert "total_latency_ms" in turn_end_msg
            assert turn_end_msg["total_latency_ms"] == turn_end_msg["total_turn_ms"]

    async def test_llm_ttft_excludes_stt_and_rag_time(self):
        """
        Verify that llm_ttft_ms strictly isolates the LLM time-to-first-token.
        STT delay: ~80ms
        RAG delay: ~50ms
        LLM TTFT delay: ~15ms
        llm_ttft_ms must be ~15ms, strictly less than STT and RAG delays.
        """
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_metrics_ttft_isolation", websocket=ws)

        async def slow_stt(audio_data, session_id=""):
            await asyncio.sleep(0.08)  # 80ms
            return "How do I order a repeat prescription?"

        async def slow_rag(query):
            await asyncio.sleep(0.05)  # 50ms
            return "Order online via the NHS App."

        async def fast_llm(history, session_id="", rag_context=None):
            await asyncio.sleep(0.015)  # 15ms
            yield "You can "
            yield "order online."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO_CHUNK"

        with patch("pipeline.transcribe_audio", side_effect=slow_stt), \
             patch("pipeline.retrieve_context", side_effect=slow_rag), \
             patch("pipeline.stream_llm_response", side_effect=fast_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            turn_end_msg = next(m for m in ws.sent_messages if m.get("type") == MSG_TURN_END)

            stt_ms = turn_end_msg["stt_ms"]
            rag_ms = turn_end_msg["rag_ms"]
            llm_ttft_ms = turn_end_msg["llm_ttft_ms"]
            ttfa_ms = turn_end_msg["speech_end_to_first_audio_ms"]
            total_ms = turn_end_msg["total_turn_ms"]

            # STT should reflect the ~80ms sleep
            assert stt_ms >= 65.0, f"stt_ms was {stt_ms}, expected >= 65"

            # RAG should reflect the ~50ms sleep
            assert rag_ms >= 40.0, f"rag_ms was {rag_ms}, expected >= 40"

            # llm_ttft_ms MUST NOT include stt_ms or rag_ms
            # If it included them, it would be >= 130ms.
            assert llm_ttft_ms < 45.0, f"llm_ttft_ms ({llm_ttft_ms}) leaked STT or RAG time!"
            assert llm_ttft_ms < stt_ms
            assert llm_ttft_ms < rag_ms

            # Time to first audio spans speech_end -> STT -> RAG -> LLM TTFT -> TTS first chunk
            assert ttfa_ms >= (stt_ms + rag_ms + llm_ttft_ms) * 0.9
            assert total_ms >= ttfa_ms

    async def test_empty_transcript_metrics_emitted_cleanly(self):
        """When STT returns an empty transcript, turn_end contains all metrics with zero/fallbacks."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_metrics_empty_transcript", websocket=ws)

        async def empty_stt(audio_data, session_id=""):
            await asyncio.sleep(0.02)
            return ""

        with patch("pipeline.transcribe_audio", side_effect=empty_stt):
            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            turn_end_msg = next(m for m in ws.sent_messages if m.get("type") == MSG_TURN_END)
            assert turn_end_msg["turn_id"] == 1
            assert turn_end_msg["stt_ms"] >= 15.0
            assert turn_end_msg["rag_ms"] == 0.0
            assert turn_end_msg["llm_ttft_ms"] == 0.0
            assert turn_end_msg["speech_end_to_first_audio_ms"] == 0.0
            assert turn_end_msg["total_turn_ms"] >= turn_end_msg["stt_ms"]

