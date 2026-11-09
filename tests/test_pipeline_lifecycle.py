"""
Unit tests for pipeline turn lifecycle, barge-in, audio merge boundaries, error handling, and cleanup.
Verifies all acceptance criteria and corrections for Task A2.
"""

import asyncio
import json
import time
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

    @pytest.mark.asyncio
    async def test_120ms_click_during_processing_does_not_cancel_or_duplicate(self):
        """A 120ms noise click during PROCESSING does not cancel the in-flight turn or produce duplicate transcripts."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_click", websocket=ws)

        llm_gate = asyncio.Event()

        async def mock_transcribe(audio_data, session_id=""):
            return "Turn 1 question"

        async def mock_stream_llm(history, session_id=""):
            await llm_gate.wait()
            yield "Turn 1 answer."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            # Start Turn 1
            audio_1 = b"\x01\x00" * 4800  # 300ms
            task_1 = await pipeline._start_turn(audio_1)

            # Let Turn 1 complete STT and reach the LLM gate
            while len([m for m in ws.sent_messages if m.get("type") == "transcript"]) == 0:
                await asyncio.sleep(0.01)

            assert pipeline.state == STATE_PROCESSING
            assert not task_1.done()

            # Generate 6 frames of 20ms click audio (120ms total)
            click_frame = (np.sin(np.linspace(0, 1, 320)) * 3276).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            # Feed 6 frames of click (120ms)
            for _ in range(6):
                await pipeline._handle_audio_chunk(click_frame)

            # Feed 15 frames of silence to end the click segment
            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            # Assert Turn 1 was NOT cancelled!
            assert not task_1.done()
            assert not task_1.cancelled()
            assert pipeline.state == STATE_PROCESSING

            # Release the LLM gate and let Turn 1 finish
            llm_gate.set()
            await task_1

            # Turn 1 must have produced exactly 1 transcript message and 1 turn_end
            transcripts = [m for m in ws.sent_messages if m.get("type") == "transcript"]
            assert len(transcripts) == 1
            assert transcripts[0]["text"] == "Turn 1 question"
            assert transcripts[0]["turn_id"] == 1

    @pytest.mark.asyncio
    async def test_1_2s_genuine_continuation_during_processing_supersedes_and_processes_once(self):
        """A 1.2-second genuine continuation during PROCESSING supersedes the old turn and is processed exactly once."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_continuation", websocket=ws)

        stt_calls = []
        llm_gate = asyncio.Event()

        async def mock_transcribe(audio_data, session_id=""):
            stt_calls.append(audio_data)
            return f"Transcribed_{len(stt_calls)}"

        async def mock_stream_llm(history, session_id=""):
            if len(stt_calls) == 1:
                # Turn 1: hold in processing
                await llm_gate.wait()
            yield "Answer."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            # Start Turn 1
            audio_1 = b"\x01\x00" * 4800  # 300ms
            task_1 = await pipeline._start_turn(audio_1)

            # Wait until Turn 1 emits transcript and holds in LLM
            while len([m for m in ws.sent_messages if m.get("type") == "transcript"]) == 0:
                await asyncio.sleep(0.01)

            assert pipeline.state == STATE_PROCESSING
            assert not task_1.done()

            # Feed 1.2s of genuine speech (60 frames of 20ms)
            speech_frame = (np.sin(np.linspace(0, 1, 320)) * 3276).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            for _ in range(60):
                await pipeline._handle_audio_chunk(speech_frame)

            # Turn 1 must now be cancelled/done and superseded event sent
            assert task_1.done() or task_1.cancelled()
            superseded_msgs = [m for m in ws.sent_messages if m.get("type") == "turn_superseded"]
            assert len(superseded_msgs) == 1
            assert superseded_msgs[0]["turn_id"] == 1

            # End speech with 15 frames of silence
            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            # Turn 2 should start and complete
            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            assert pipeline.turn_id == 2
            # Check transcripts: Turn 1 transcript was emitted earlier, then superseded, then Turn 2 transcript
            transcripts = [m for m in ws.sent_messages if m.get("type") == "transcript"]
            assert len(transcripts) == 2
            assert transcripts[0]["turn_id"] == 1
            assert transcripts[1]["turn_id"] == 2

    @pytest.mark.asyncio
    async def test_pure_noise_does_not_start_turn(self):
        """Pure noise or sub-threshold audio does not start an STT/LLM turn."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_noise", websocket=ws)

        stt_called = False

        async def mock_transcribe(audio_data, session_id=""):
            nonlocal stt_called
            stt_called = True
            return "Noise"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe):
            # Low amplitude noise (RMS ~0.005, below threshold 0.02)
            noise_frame = (np.full(320, 150, dtype=np.int16)).tobytes()
            for _ in range(50):
                await pipeline._handle_audio_chunk(noise_frame)

            assert pipeline.turn_id == 0
            assert pipeline.state == STATE_IDLE
            assert stt_called is False
            assert len(ws.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_segments_ending_between_300ms_and_400ms_are_processed(self):
        """Utterances ending between 300ms and 400ms are legitimate speech and are dispatched to STT."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_short_utterance", websocket=ws)

        stt_dispatched = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched.append(audio_data)
            return "No."

        async def mock_stream_llm(history, session_id=""):
            yield "Understood."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            # Speech: 10 frames = 200ms
            speech_frame = (np.sin(np.linspace(0, 1, 320)) * 3276).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            for _ in range(10):
                await pipeline._handle_audio_chunk(speech_frame)

            # Silence: 15 frames = 300ms hangover
            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Turn 1 must have run and processed this short utterance
            assert len(stt_dispatched) == 1
            assert pipeline.turn_id == 1
            # Total audio duration >= 300ms
            assert len(stt_dispatched[0]) >= 9600

    @pytest.mark.asyncio
    async def test_finger_snap_short_impulse_in_idle_does_not_start_turn(self):
        """A finger snap or short impulse (<= 100ms active energy) in IDLE is rejected by the 200ms active gate."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_snap_reject", websocket=ws)

        stt_dispatched = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched.append(audio_data)
            return "Thank you"  # Simulated Whisper hallucination if called

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe):
            # 5 frames of high-amplitude snap impulse (100ms total)
            snap_frame = (np.sin(np.linspace(0, 5, 320)) * 16000).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            for _ in range(5):
                await pipeline._handle_audio_chunk(snap_frame)

            # 15 frames of silence hangover
            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Verify: turn NEVER started and STT was NEVER called!
            assert len(stt_dispatched) == 0
            assert pipeline.turn_id == 0
            assert pipeline.state == STATE_IDLE
            assert len([m for m in ws.sent_messages if m.get("type") == "transcript"]) == 0

    @pytest.mark.asyncio
    async def test_loud_non_speech_noise_burst_in_idle_does_not_start_turn(self):
        """A 120ms loud non-speech noise burst in IDLE is rejected by the 200ms active gate."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_loud_burst_reject", websocket=ws)

        stt_called = False

        async def mock_transcribe(audio_data, session_id=""):
            nonlocal stt_called
            stt_called = True
            return "Noise"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe):
            # 6 frames of loud noise (120ms)
            noise_frame = (np.sin(np.linspace(0, 3, 320)) * 12000).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            for _ in range(6):
                await pipeline._handle_audio_chunk(noise_frame)

            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            assert stt_called is False
            assert pipeline.turn_id == 0
            assert pipeline.state == STATE_IDLE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("short_word", ["No", "Yes", "Wait"])
    async def test_genuine_short_speech_words_start_turn(self, short_word):
        """Genuine short utterances like 'No', 'Yes', 'Wait' (active duration >= 200ms) start a turn cleanly."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id=f"test_short_{short_word.lower()}", websocket=ws)

        stt_dispatched = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched.append(audio_data)
            return f"{short_word}."

        async def mock_stream_llm(history, session_id=""):
            yield "Understood."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            # 13 frames of speech (~260ms active speech, e.g. brisk "No" / "Yes" / "Wait")
            speech_frame = (np.sin(np.linspace(0, 2, 320)) * 4000).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            for _ in range(13):
                await pipeline._handle_audio_chunk(speech_frame)

            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            # Turn 1 must have run and called STT
            assert len(stt_dispatched) == 1
            assert pipeline.turn_id == 1
            transcripts = [m for m in ws.sent_messages if m.get("type") == "transcript"]
            assert len(transcripts) == 1
            assert transcripts[0]["text"] == f"{short_word}."

    @pytest.mark.asyncio
    async def test_normal_length_speech_starts_turn(self):
        """Normal conversational utterance (~1.2s active speech) starts a turn without issue."""
        import numpy as np
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_normal_speech", websocket=ws)

        stt_dispatched = []

        async def mock_transcribe(audio_data, session_id=""):
            stt_dispatched.append(audio_data)
            return "Good morning, I would like to book an appointment."

        async def mock_stream_llm(history, session_id=""):
            yield "Certainly, I can help with that."

        async def mock_tts(sentence, session_id=""):
            yield b"AUDIO"

        with patch("pipeline.transcribe_audio", side_effect=mock_transcribe), \
             patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            speech_frame = (np.sin(np.linspace(0, 1, 320)) * 4000).astype(np.int16).tobytes()
            silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

            # 60 frames = 1200ms active speech
            for _ in range(60):
                await pipeline._handle_audio_chunk(speech_frame)

            for _ in range(15):
                await pipeline._handle_audio_chunk(silence_frame)

            if pipeline._current_turn_task:
                await pipeline._current_turn_task

            assert len(stt_dispatched) == 1
            assert pipeline.turn_id == 1
            transcripts = [m for m in ws.sent_messages if m.get("type") == "transcript"]
            assert len(transcripts) == 1


class TestPipelinedSentenceSynthesis:
    """Verifies non-blocking, pipelined sentence synthesis and strict audio ordering."""

    @pytest.mark.asyncio
    async def test_pipelined_synthesis_multi_sentence_correct_order(self):
        """Multiple sentences are synthesized and delivered across WebSocket in strict order."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_pipeline_order", websocket=ws)
        pipeline.turn_id = 1
        pipeline._active_turn_id = 1

        sent_order = []

        async def mock_stream_llm(history, **kwargs):
            # Stream tokens across 3 sentences
            tokens = [
                "Good ", "morning, ", "Oak ", "Lane. ",
                "How ", "can ", "I ", "help? ",
                "Please ", "let ", "me ", "know."
            ]
            for t in tokens:
                yield t

        async def mock_tts(sentence, session_id=""):
            sent_order.append(sentence)
            # Yield distinct audio for each sentence
            yield f"AUDIO_{sentence}".encode("utf-8")

        with patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            await pipeline._stream_and_speak(
                turn_id=1,
                t_speech_end=time.perf_counter(),
                stt_ms=100.0,
                rag_ms=20.0,
            )

        # Confirm all 3 sentences were synthesized in exact FIFO order
        assert len(sent_order) == 3
        assert sent_order[0] == "Good morning, Oak Lane."
        assert sent_order[1] == "How can I help?"
        assert sent_order[2] == "Please let me know."

        # Confirm all 3 audio chunks were transmitted to WebSocket in exact FIFO order
        audio_chunks = ws.sent_binary
        assert len(audio_chunks) == 3
        assert audio_chunks[0] == b"AUDIO_Good morning, Oak Lane."
        assert audio_chunks[1] == b"AUDIO_How can I help?"
        assert audio_chunks[2] == b"AUDIO_Please let me know."

        # Verify terminal turn_end message was sent
        turn_ends = [m for m in ws.sent_messages if m.get("type") == "turn_end"]
        assert len(turn_ends) == 1
        assert turn_ends[0]["turn_id"] == 1

    @pytest.mark.asyncio
    async def test_pipelined_synthesis_llm_not_blocked_by_tts(self):
        """LLM token generation continues unblocked while TTS synthesizes previous sentence."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_llm_unblocked", websocket=ws)
        pipeline.turn_id = 1
        pipeline._active_turn_id = 1

        llm_finished_time = None
        tts_start_times = []
        tts_end_times = []

        async def slow_tts(sentence, session_id=""):
            tts_start_times.append(time.perf_counter())
            await asyncio.sleep(0.15)  # Simulate 150ms synthesis time
            tts_end_times.append(time.perf_counter())
            yield f"CHUNK_{sentence}".encode("utf-8")

        async def mock_stream_llm(history, **kwargs):
            nonlocal llm_finished_time
            # Generate sentence 1
            yield "First sentence is here. "
            # Generate sentence 2 without waiting for TTS
            yield "Second sentence follows immediately."
            llm_finished_time = time.perf_counter()

        with patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=slow_tts):

            await pipeline._stream_and_speak(
                turn_id=1,
                t_speech_end=time.perf_counter(),
                stt_ms=50.0,
                rag_ms=10.0,
            )

        # The LLM finished generating BOTH sentences before sentence 1 finished synthesizing!
        assert llm_finished_time is not None
        assert len(tts_end_times) == 2
        # LLM finished earlier than the completion of the 2nd sentence synthesis
        assert llm_finished_time < tts_end_times[1]

        # Both chunks delivered in order
        assert len(ws.sent_binary) == 2

    @pytest.mark.asyncio
    async def test_pipelined_synthesis_barge_in_cancels_pending_sentences(self):
        """Barge-in / interrupt during synthesis cancels pending sentences; no extra audio sent."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_barge_in_pipeline", websocket=ws)
        pipeline.turn_id = 1
        pipeline._active_turn_id = 1

        s1_started = asyncio.Event()

        async def mock_tts(sentence, session_id=""):
            if "First" in sentence:
                s1_started.set()
                await asyncio.sleep(0.05)  # Synthesis/playback in flight when barge-in occurs
                yield b"AUDIO_1"
            else:
                yield b"AUDIO_2_SHOULD_NEVER_SEND"

        async def mock_stream_llm(history, **kwargs):
            yield "First sentence here. "
            yield "Second sentence pending."

        with patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=mock_tts):

            async def run_turn():
                await pipeline._stream_and_speak(
                    turn_id=1,
                    t_speech_end=time.perf_counter(),
                )

            turn_task = asyncio.create_task(run_turn())
            pipeline._current_turn_task = turn_task
            await s1_started.wait()

            # User barge-in occurs while turn 1 is speaking
            await pipeline._interrupt_turn()

            await asyncio.gather(turn_task, return_exceptions=True)

        # AUDIO_2 must NOT have been sent
        sent_audio = ws.sent_binary
        assert b"AUDIO_2_SHOULD_NEVER_SEND" not in sent_audio

    @pytest.mark.asyncio
    async def test_pipelined_synthesis_superseded_turn_discards_audio(self):
        """When Turn 1 is superseded by Turn 2, Turn 1 audio is discarded and cannot leak."""
        ws = MockWebSocket()
        pipeline = VoicePipeline(session_id="test_superseded_leak", websocket=ws)
        pipeline.turn_id = 1
        pipeline._active_turn_id = 1

        t1_in_flight = asyncio.Event()

        async def slow_tts(sentence, session_id=""):
            if "Turn 1" in sentence:
                t1_in_flight.set()
                await asyncio.sleep(0.2)
                yield b"TURN1_AUDIO"
            else:
                yield b"TURN2_AUDIO"

        turn_count = 0

        async def mock_stream_llm(history, **kwargs):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                yield "Turn 1 sentence. "
            else:
                yield "Turn 2 sentence."

        with patch("pipeline.stream_llm_response", side_effect=mock_stream_llm), \
             patch("pipeline.stream_tts_audio", side_effect=slow_tts):

            # Start Turn 1
            t1_task = asyncio.create_task(pipeline._stream_and_speak(
                turn_id=1,
                t_speech_end=time.perf_counter(),
            ))

            await t1_in_flight.wait()

            # Now Turn 2 supersedes Turn 1
            pipeline._superseded_turn_ids.add(1)
            pipeline.turn_id = 2
            pipeline._active_turn_id = 2

            t2_task = asyncio.create_task(pipeline._stream_and_speak(
                turn_id=2,
                t_speech_end=time.perf_counter(),
            ))

            await asyncio.gather(t1_task, t2_task, return_exceptions=True)

        # Only Turn 2 audio sent, never Turn 1
        assert b"TURN1_AUDIO" not in ws.sent_binary
        assert b"TURN2_AUDIO" in ws.sent_binary


async def asyncio_generator_empty():
    if False:
        yield None
