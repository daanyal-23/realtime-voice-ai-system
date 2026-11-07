"""
Unit tests for pipeline turn lifecycle, barge-in, audio merge boundaries, error handling, and cleanup.
Verifies all acceptance criteria and corrections for Task A2.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import LLMError
from pipeline import (
    VoicePipeline,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_PROCESSING,
    STATE_SPEAKING,
    APOLOGY_MESSAGE,
    INTERNAL_ERROR_PAYLOAD,
    MAX_MERGE_BYTES,
)


class MockWebSocket:
    """Mock WebSocket for capturing sent control and binary messages."""

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


class TestPipelineLifecycle:

    @pytest.mark.asyncio
    async def test_starting_second_turn_cancels_and_awaits_first(self):
        """Calling _start_turn() while a turn is active cancels and awaits the first turn."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_cancel_await", websocket=ws)

        turn1_entered = asyncio.Event()
        turn1_cancelled = False
        turn1_unwound = False

        audio_1 = b"\x01\x00" * 4800  # 9600 bytes
        audio_2 = b"\x02\x00" * 4800

        async def mock_transcribe(audio_data, session_id=""):
            nonlocal turn1_cancelled, turn1_unwound
            if audio_data == audio_1:
                turn1_entered.set()
                try:
                    # Suspend Turn 1 until cancelled
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    turn1_cancelled = True
                    turn1_unwound = True
                    raise
            else:
                # Turn 2 STT returns immediately so Turn 2 does not hang
                return "turn 2 transcript"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", return_value=asyncio_generator_empty()), \
             patch("pipeline.stream_tts_audio", return_value=asyncio_generator_empty()):

            # Start Turn 1
            task_1 = await pipeline._start_turn(audio_1)
            assert pipeline.turn_id == 1
            assert pipeline._active_turn_id == 1
            assert not task_1.done()

            # Wait until Turn 1 has entered its mocked STT coroutine
            await turn1_entered.wait()

            # Now start Turn 2 via _start_turn
            task_2 = await pipeline._start_turn(audio_2)

            # Turn 1 must be cancelled AND unwound/awaited
            assert turn1_cancelled is True
            assert turn1_unwound is True
            assert task_1.done()
            assert pipeline.turn_id == 2
            assert pipeline._active_turn_id == 2
            assert pipeline._current_turn_task is task_2

            # Ensure Turn 2 finishes without hanging
            await task_2
            assert task_2.done()

            # Clean up turn 2
            await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_old_turn_cannot_reset_new_turn_state(self):
        """Production _execute_turn() cleanup in finally block cannot reset the new turn's state to IDLE."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_state_isolation", websocket=ws)

        turn1_started = asyncio.Event()

        async def suspend_stt(audio_data, session_id=""):
            turn1_started.set()
            # Suspend inside STT until cancelled
            await asyncio.Event().wait()

        with patch("pipeline.transcribe_audio", side_effect=suspend_stt):
            # 1. Start Turn 1 using actual production _execute_turn
            pipeline.turn_id = 1
            pipeline._active_turn_id = 1
            pipeline.state = STATE_PROCESSING

            turn1_task = asyncio.create_task(pipeline._execute_turn(1, b"audio_turn_1"))
            pipeline._current_turn_task = turn1_task

            # 2. Suspend Turn 1 inside mocked STT
            await asyncio.wait_for(turn1_started.wait(), timeout=2.0)
            assert not turn1_task.done()

            # 3. Simulate Turn 2 becoming the active turn and reaching SPEAKING
            pipeline.turn_id = 2
            pipeline._active_turn_id = 2
            pipeline.state = STATE_SPEAKING

            # 4. Cancel and await Turn 1
            turn1_task.cancel()
            try:
                await asyncio.wait_for(turn1_task, timeout=2.0)
            except asyncio.CancelledError:
                pass

            # 5. Verify the actual production finally block does not reset Turn 2's state to IDLE
            assert pipeline.state == STATE_SPEAKING
            assert pipeline._active_turn_id == 2

    @pytest.mark.asyncio
    async def test_merged_audio_14_9s_accepted(self):
        """A 14.9-second merged audio segment is accepted (< 15.0s strictly)."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_merge_14_9", websocket=ws)

        stt_dispatched_audio = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched_audio.append(audio_data)
            return "ok"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", return_value=asyncio_generator_empty()), \
             patch("pipeline.stream_tts_audio", return_value=asyncio_generator_empty()):

            # 14.9s = 14.9 * 32000 = 476,800 bytes
            prev_audio = b"\x01\x00" * 100000  # 200,000 bytes (~6.25s)
            raw_audio = b"\x02\x00" * 138400   # 276,800 bytes (~8.65s)
            # Total candidate = 476,800 bytes (14.9s < 15.0s)

            pipeline.state = STATE_PROCESSING
            pipeline._last_speech_end_time = 100.0
            pipeline._last_processed_audio = prev_audio

            # User speaks again within 0.5s gap (< 2s)
            with patch("time.perf_counter", return_value=100.5), \
                 patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x00" * 640)

            assert pipeline._pending_merge_audio == prev_audio

            # Second speech ends with raw_audio (preloading buffer minus final 640-byte chunk)
            pipeline._audio_buffer = bytearray(raw_audio[:-640])
            with patch.object(pipeline.vad, "process", return_value=False):
                await pipeline._handle_audio_chunk(raw_audio[-640:])

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Merged audio of exactly 476,800 bytes was dispatched to STT
            assert len(stt_dispatched_audio) == 1
            assert len(stt_dispatched_audio[0]) == 476800
            assert stt_dispatched_audio[0] == prev_audio + raw_audio

    @pytest.mark.asyncio
    async def test_merged_audio_15_0s_rejected(self):
        """A 15.0-second merged segment reaches the 15s limit and is rejected (condition is strictly < 15s)."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_merge_15_0", websocket=ws)

        stt_dispatched_audio = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched_audio.append(audio_data)
            return "ok"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", return_value=asyncio_generator_empty()), \
             patch("pipeline.stream_tts_audio", return_value=asyncio_generator_empty()):

            # 15.0s = 15.0 * 32000 = 480,000 bytes exactly
            prev_audio = b"\x01\x00" * 120000  # 240,000 bytes (7.5s)
            raw_audio = b"\x02\x00" * 120000   # 240,000 bytes (7.5s)
            # Total candidate = 480,000 bytes (== MAX_MERGE_BYTES)

            pipeline.state = STATE_PROCESSING
            pipeline._last_speech_end_time = 100.0
            pipeline._last_processed_audio = prev_audio

            with patch("time.perf_counter", return_value=100.5), \
                 patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x00" * 640)

            assert pipeline._pending_merge_audio == prev_audio

            # Second speech ends with raw_audio (preloading buffer minus final 640-byte chunk)
            pipeline._audio_buffer = bytearray(raw_audio[:-640])
            with patch.object(pipeline.vad, "process", return_value=False):
                await pipeline._handle_audio_chunk(raw_audio[-640:])

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Merge rejected: only raw_audio (240,000 bytes) was dispatched as a standalone turn
            assert len(stt_dispatched_audio) == 1
            assert len(stt_dispatched_audio[0]) == 240000
            assert stt_dispatched_audio[0] == raw_audio

    @pytest.mark.asyncio
    async def test_merged_audio_exceeding_15s_rejected(self):
        """A merged segment exceeding 15.0s (e.g. 16.0s = 512,000 bytes) is rejected."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_merge_16_0", websocket=ws)

        stt_dispatched_audio = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched_audio.append(audio_data)
            return "ok"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", return_value=asyncio_generator_empty()), \
             patch("pipeline.stream_tts_audio", return_value=asyncio_generator_empty()):

            # 16.0s = 512,000 bytes
            prev_audio = b"\x01\x00" * 128000  # 256,000 bytes (8s)
            raw_audio = b"\x02\x00" * 128000   # 256,000 bytes (8s)

            pipeline.state = STATE_PROCESSING
            pipeline._last_speech_end_time = 100.0
            pipeline._last_processed_audio = prev_audio

            with patch("time.perf_counter", return_value=100.5), \
                 patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x00" * 640)

            pipeline._audio_buffer = bytearray(raw_audio[:-640])
            with patch.object(pipeline.vad, "process", return_value=False):
                await pipeline._handle_audio_chunk(raw_audio[-640:])

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Discarded merge: standalone raw_audio received by STT
            assert len(stt_dispatched_audio) == 1
            assert len(stt_dispatched_audio[0]) == 256000
            assert stt_dispatched_audio[0] == raw_audio

    @pytest.mark.asyncio
    async def test_new_standalone_turn_receives_correct_audio(self):
        """When merge is rejected by gap >= 2.0s, standalone turn receives only the new utterance."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_gap_reject", websocket=ws)

        stt_dispatched_audio = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched_audio.append(audio_data)
            return "ok"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", return_value=asyncio_generator_empty()), \
             patch("pipeline.stream_tts_audio", return_value=asyncio_generator_empty()):

            prev_audio = b"\x01\x00" * 8000  # 16,000 bytes (> 300ms threshold)
            raw_audio = b"\x02\x00" * 8000   # 16,000 bytes (> 300ms threshold)

            pipeline.state = STATE_PROCESSING
            pipeline._last_speech_end_time = 100.0
            pipeline._last_processed_audio = prev_audio

            # Gap is 2.5s >= 2.0s
            with patch("time.perf_counter", return_value=102.5), \
                 patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x00" * 640)

            # Gap exceeded -> merge was not flagged
            assert pipeline._pending_merge_audio is None

            pipeline._audio_buffer = bytearray(raw_audio[:-640])
            with patch.object(pipeline.vad, "process", return_value=False):
                await pipeline._handle_audio_chunk(raw_audio[-640:])

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            assert len(stt_dispatched_audio) == 1
            assert stt_dispatched_audio[0] == raw_audio

    @pytest.mark.asyncio
    async def test_exactly_one_terminal_event_per_turn(self):
        """Turn-specific tracking ensures exactly one of turn_end, interrupted, or error per turn."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_single_terminal", websocket=ws)

        async def mock_transcribe(audio_data, session_id=""):
            return "Query"

        async def mock_stream_llm(history, session_id=""):
            yield "Answer."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            # Turn 1: normal run
            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            # Attempt duplicate terminal calls for Turn 1
            assert pipeline._is_terminal_sent(1) is True
            assert pipeline._mark_terminal_sent(1) is False
            await pipeline._send_sanitized_error(turn_id=1)

            # Check messages for Turn 1: exactly one terminal event (turn_end)
            t1_terminals = [
                m for m in ws.sent_messages
                if m.get("turn_id") == 1 and m.get("type") in ("turn_end", "interrupted", "error")
            ]
            assert len(t1_terminals) == 1
            assert t1_terminals[0]["type"] == "turn_end"

    @pytest.mark.asyncio
    async def test_second_utterance_during_processing_cancels_and_merges_audio(self):
        """Second utterance during PROCESSING cancels the first turn, merges audio, and makes 1 LLM call."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_merge", websocket=ws)

        stt_calls = []
        llm_calls = []
        stt_gate = asyncio.Event()

        async def mock_transcribe(audio_data, session_id=""):
            stt_calls.append(audio_data)
            await stt_gate.wait()
            return f"transcribed_{len(audio_data)}_bytes"

        async def mock_stream_llm(history, session_id=""):
            llm_calls.append(history)
            yield "Hello"

        async def mock_tts(sentence, session_id=""):
            yield b"RIFF_WAV_BYTES"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            audio_1 = b"\x01\x00" * 4800
            pipeline.state = STATE_IDLE
            pipeline._last_speech_end_time = 100.0
            pipeline._last_processed_audio = audio_1
            task_1 = await pipeline._start_turn(audio_1)

            assert pipeline.state == STATE_PROCESSING
            assert pipeline.turn_id == 1

            # Speech start within 0.5s gap (< 2s)
            with patch("time.perf_counter", return_value=100.5), \
                 patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x02\x00" * 320)

            assert pipeline.state == STATE_LISTENING
            assert task_1.cancelled() or task_1.done()
            assert pipeline._pending_merge_audio == audio_1
            assert len(pipeline.conversation_history) == 0

            stt_gate.set()

            # Second speech ends (preload buffer minus final 640-byte chunk)
            audio_2 = b"\x03\x00" * 4800
            pipeline._audio_buffer = bytearray(audio_2[:-640])
            with patch.object(pipeline.vad, "process", return_value=False):
                await pipeline._handle_audio_chunk(audio_2[-640:])

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            assert len(llm_calls) == 1
            assert any(call == audio_1 + audio_2 for call in stt_calls)
            assert len(pipeline.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_barge_in_during_speaking_cancels_tts_and_sends_interrupted(self):
        """Speech while SPEAKING triggers barge-in: cancels TTS, sends interrupted, starts new standalone turn."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_bargein", websocket=ws)

        tts_cancelled = False

        def mock_cancel_tts():
            nonlocal tts_cancelled
            tts_cancelled = True

        with patch("pipeline.cancel_current_tts", side_effect=mock_cancel_tts):
            pipeline.state = STATE_SPEAKING
            pipeline.turn_id = 5
            pipeline._active_turn_id = 5

            with patch.object(pipeline.vad, "process", return_value=True):
                await pipeline._handle_audio_chunk(b"\x05\x00" * 320)

            assert tts_cancelled is True
            assert pipeline.state == STATE_LISTENING
            assert pipeline._pending_merge_audio is None

            interrupted_msgs = [m for m in ws.sent_messages if m.get("type") == "interrupted"]
            assert len(interrupted_msgs) == 1
            assert interrupted_msgs[0]["turn_id"] == 5

    @pytest.mark.asyncio
    async def test_llm_error_path_speaks_apology_and_cleans_history(self):
        """When LLMError occurs, fixed apology is spoken, turn ends, and orphan user message is removed."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_llmerror", websocket=ws)

        spoken_sentences = []

        async def mock_transcribe(audio_data, session_id=""):
            return "Please help me with an appointment"

        async def mock_stream_llm(history, session_id=""):
            raise LLMError("Both primary and fallback models failed")
            yield "never"

        async def mock_tts(sentence, session_id=""):
            spoken_sentences.append(sentence)
            yield b"APOLOGY_AUDIO_WAV"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            audio = b"\x01\x00" * 4800
            await pipeline._start_turn(audio)
            await pipeline._current_turn_task

            assert APOLOGY_MESSAGE in spoken_sentences
            assert len(pipeline.conversation_history) == 0
            turn_ends = [m for m in ws.sent_messages if m.get("type") == "turn_end"]
            assert len(turn_ends) == 1
            assert pipeline.state == STATE_IDLE

    @pytest.mark.asyncio
    async def test_disconnect_cleanup_leaves_no_pending_tasks(self):
        """cleanup() on disconnect cancels and awaits the turn task with zero pending session tasks."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_cleanup", websocket=ws)

        turn_started = asyncio.Event()
        turn_cancelled = False

        async def long_running_turn(turn_id, audio):
            turn_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                nonlocal turn_cancelled
                turn_cancelled = True
                raise

        pipeline._current_turn_task = asyncio.create_task(long_running_turn(1, b"audio"))
        pipeline.state = STATE_SPEAKING

        # Await that the task has started before calling cleanup
        await asyncio.wait_for(turn_started.wait(), timeout=2.0)

        await pipeline.cleanup()

        assert turn_cancelled is True
        assert pipeline._current_turn_task is None
        assert pipeline.state == STATE_IDLE

    @pytest.mark.asyncio
    async def test_error_payload_has_no_exception_text(self):
        """Unexpected internal exceptions send sanitized internal_error payload with no leak of exception text."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_error", websocket=ws)

        secret_error_text = "SensitiveDBPasswordOrInternalExceptionText12345"

        async def failing_transcribe(audio_data, session_id=""):
            raise RuntimeError(secret_error_text)

        with patch("pipeline.transcribe_audio", side_effect=failing_transcribe):
            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            error_msgs = [m for m in ws.sent_messages if m.get("type") == "error"]
            assert len(error_msgs) == 1
            payload = error_msgs[0]
            assert payload["code"] == "internal_error"
            assert payload["message"] == "Something went wrong. Please try again."
            payload_str = json.dumps(payload)
            assert secret_error_text not in payload_str
            assert "RuntimeError" not in payload_str
            assert pipeline.state == STATE_IDLE

    @pytest.mark.asyncio
    async def test_whitespace_normalization_before_tts(self):
        """Sentences passed to TTS are normalized with ' '.join(s.split())."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_norm", websocket=ws)

        tts_inputs = []

        async def mock_transcribe(audio_data, session_id=""):
            return "Test query"

        async def mock_stream_llm(history, session_id=""):
            yield "Hello   there. \n\n"
            yield " How \t are \n you doing? "

        async def mock_tts(sentence, session_id=""):
            tts_inputs.append(sentence)
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            await pipeline._start_turn(b"\x01\x00" * 4800)
            await pipeline._current_turn_task

            assert "Hello there." in tts_inputs
            assert "How are you doing?" in tts_inputs
            assert not any("  " in s or "\n" in s or "\t" in s for s in tts_inputs)


async def asyncio_generator_empty():
    if False:
        yield None
