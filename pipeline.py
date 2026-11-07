"""
VoicePipeline: Orchestrates the full STT → LLM → TTS loop with explicit turn lifecycle.

States:
  IDLE: Waiting for user speech.
  LISTENING: User is actively speaking (VAD active).
  PROCESSING: Speech ended; running STT, RAG, and LLM until first audio sent.
  SPEAKING: First audio sent; audio playback until turn_end.

Lifecycle guarantees:
  - Incrementing turn_id for every turn.
  - Exactly one active turn task at a time, managed via async _start_turn().
  - Previous turn is cancelled and awaited before a new turn starts.
  - Cancellation reaches STT, LLM generator (always aclosed), and TTS.
  - Speech during SPEAKING: barge-in (cancel, send "interrupted", standalone new turn).
  - Speech during PROCESSING: cancel turn; merge audio if gap < 2s and total < 15s (strictly < 15.0s).
  - If merged audio reaches or exceeds 15.0s, merge is discarded and new utterance runs as standalone turn.
  - Every turn ends with exactly one terminal event: turn_end, interrupted, or error.
  - LLMError speaks fixed apology line, ends turn, and leaves no orphan user message.
  - Sentences normalized with " ".join(s.split()) before TTS.
  - Client errors sanitized to {"type": "error", "code": "internal_error", "message": "Something went wrong. Please try again."}.
  - On disconnect: cancel and await turn task.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Set

# pyrefly: ignore [missing-import]
import numpy as np

from stt import transcribe_audio
from llm import stream_llm_response, LLMError
from tts import stream_tts_audio, cancel_current_tts
from vad import VAD

logger = logging.getLogger("pipeline")

# Protocol message types (sent to client)
MSG_TRANSCRIPT = "transcript"         # user's speech text
MSG_AI_TEXT = "ai_text"               # AI response text chunk
MSG_AUDIO_CHUNK = "audio_chunk"       # base64 TTS audio chunk
MSG_TURN_START = "turn_start"         # AI started speaking
MSG_TURN_END = "turn_end"             # AI finished speaking
MSG_INTERRUPTED = "interrupted"       # AI was interrupted
MSG_ERROR = "error"                   # error notification
MSG_STATUS = "status"                 # system status

# Pipeline states
STATE_IDLE = "IDLE"
STATE_LISTENING = "LISTENING"
STATE_PROCESSING = "PROCESSING"
STATE_SPEAKING = "SPEAKING"

# Timing and audio constraints
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit mono PCM
MAX_MERGE_GAP_S = 2.0
MAX_MERGE_TOTAL_S = 15.0
MAX_MERGE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_MERGE_TOTAL_S)  # 480,000 bytes (strictly < 15s)
MIN_SPEECH_BYTES = 9600  # 300ms at 16kHz 16-bit
MAX_SPEECH_BYTES = 16000 * 2 * 8  # 8 seconds max per speech buffer chunk

APOLOGY_MESSAGE = "Sorry, I'm having a technical problem. Please try again in a moment."
INTERNAL_ERROR_PAYLOAD = {
    "type": "error",
    "code": "internal_error",
    "message": "Something went wrong. Please try again.",
}


class VoicePipeline:
    def __init__(self, session_id: str, websocket):
        self.session_id = session_id
        self.websocket = websocket
        self.vad = VAD()

        # Conversation history for LLM context
        self.conversation_history = []

        # Turn lifecycle state
        self.state = STATE_IDLE
        self.turn_id = 0
        self._active_turn_id: Optional[int] = None
        self._current_turn_task: Optional[asyncio.Task] = None
        self._active_llm_gen = None

        # Turn-specific terminal event tracking
        self._terminated_turn_ids: Set[int] = set()

        # Interruption control
        self._interrupt_event = asyncio.Event()

        # Audio buffering and merge control
        self._audio_buffer = bytearray()
        self._speech_active = False
        self._last_speech_end_time: Optional[float] = None
        self._last_processed_audio: Optional[bytes] = None
        self._pending_merge_audio: Optional[bytes] = None

        logger.info(f"[{self.session_id}] Pipeline initialized (state={self.state})")

    def _mark_terminal_sent(self, turn_id: int) -> bool:
        """
        Record that a terminal event was sent for turn_id.
        Returns True if this call marked it (first terminal event),
        or False if a terminal event was already sent for this turn.
        """
        if turn_id in self._terminated_turn_ids:
            return False
        self._terminated_turn_ids.add(turn_id)
        return True

    def _is_terminal_sent(self, turn_id: int) -> bool:
        """Check whether a terminal event was already sent for turn_id."""
        return turn_id in self._terminated_turn_ids

    async def run(self, websocket):
        """Main loop: receive audio/control messages from client."""
        await self._send(MSG_STATUS, {"message": "connected", "session_id": self.session_id})

        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    await self._handle_audio_chunk(message)
                elif isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_control(data)
            except json.JSONDecodeError:
                logger.warning(f"[{self.session_id}] Invalid JSON control message")
            except Exception as e:
                logger.exception(f"[{self.session_id}] Error handling message: {e}")
                await self._send_sanitized_error()

    async def _handle_audio_chunk(self, chunk: bytes):
        """
        Process incoming PCM audio chunk through VAD.
        16kHz, 16-bit mono PCM expected.
        """
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        speech_detected = self.vad.process(audio_np)

        if speech_detected and not self._speech_active:
            # Speech just started
            self._speech_active = True
            self._audio_buffer = bytearray()
            logger.info(f"[{self.session_id}] Speech start detected (current state={self.state})")

            if self.state == STATE_SPEAKING:
                # Barge-in: user spoke while AI was speaking
                logger.info(f"[{self.session_id}] ⚡ Barge-in detected during SPEAKING")
                self._pending_merge_audio = None
                await self._interrupt_turn()
                self.state = STATE_LISTENING

            elif self.state == STATE_PROCESSING:
                # Speech detected while previous turn is still PROCESSING (STT/LLM setup)
                now = time.perf_counter()
                gap = (now - self._last_speech_end_time) if self._last_speech_end_time else 999.0
                prev_audio = self._last_processed_audio or b""

                if gap < MAX_MERGE_GAP_S:
                    logger.info(
                        f"[{self.session_id}] Speech during PROCESSING within gap ({gap:.2f}s < 2s). Flagging audio merge."
                    )
                    self._pending_merge_audio = prev_audio
                else:
                    logger.info(
                        f"[{self.session_id}] Speech during PROCESSING exceeded gap ({gap:.2f}s >= 2s). Starting fresh turn."
                    )
                    self._pending_merge_audio = None

                # Cancel previous turn; leaves nothing in conversation history
                await self._cancel_current_turn()
                self.state = STATE_LISTENING

            else:
                self.state = STATE_LISTENING

        if self._speech_active:
            self._audio_buffer.extend(chunk)

        if (not speech_detected and self._speech_active) or \
           (self._speech_active and len(self._audio_buffer) > MAX_SPEECH_BYTES):
            # Speech just ended
            self._speech_active = False
            raw_audio = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()

            # Enforce 15-second limit strictly against the complete merged audio immediately before STT dispatch
            combined_audio = raw_audio
            if self._pending_merge_audio:
                candidate_merged = self._pending_merge_audio + raw_audio
                self._pending_merge_audio = None
                if len(candidate_merged) < MAX_MERGE_BYTES:
                    combined_audio = candidate_merged
                    logger.info(
                        f"[{self.session_id}] Merged audio segments: {len(combined_audio)} bytes (< 15.0s limit)"
                    )
                else:
                    # Reaches or exceeds 15 seconds: discard merge and process new utterance as standalone turn
                    logger.info(
                        f"[{self.session_id}] Merged audio ({len(candidate_merged)} bytes) >= 15.0s limit. "
                        f"Discarding merge and processing new utterance as standalone turn ({len(raw_audio)} bytes)."
                    )
                    combined_audio = raw_audio

            # Check minimum duration threshold (300ms)
            if len(combined_audio) > MIN_SPEECH_BYTES:
                self._last_speech_end_time = time.perf_counter()
                self._last_processed_audio = combined_audio
                logger.info(f"[{self.session_id}] Speech end detected ({len(combined_audio)} bytes). Starting turn.")
                await self._start_turn(combined_audio)
            else:
                logger.debug(f"[{self.session_id}] Audio segment too short ({len(combined_audio)} bytes), ignoring.")
                if self.state == STATE_LISTENING:
                    self.state = STATE_IDLE

    async def _handle_control(self, data: dict):
        """Handle JSON control messages from client."""
        msg_type = data.get("type")

        if msg_type == "interrupt":
            logger.info(f"[{self.session_id}] Client sent interrupt control message")
            await self._interrupt_turn()
            self.state = STATE_IDLE

        elif msg_type == "reset":
            await self._cancel_current_turn()
            self.conversation_history = []
            self.state = STATE_IDLE
            await self._send(MSG_STATUS, {"message": "conversation reset"})

        elif msg_type == "ping":
            await self._send("pong", {"ts": time.time()})

    async def _start_turn(self, audio_data: bytes) -> asyncio.Task:
        """
        Create a new turn task, cancelling and awaiting any existing turn.
        Guarantees strictly one active turn task per session.
        """
        await self._cancel_current_turn()

        self.turn_id += 1
        turn_id = self.turn_id
        self._active_turn_id = turn_id
        self.state = STATE_PROCESSING
        self._interrupt_event.clear()

        # Fire and manage the turn task
        self._current_turn_task = asyncio.create_task(
            self._execute_turn(turn_id, audio_data)
        )
        return self._current_turn_task

    async def _cancel_current_turn(self):
        """Cancel and await the currently active turn task, stopping TTS and LLM."""
        self._interrupt_event.set()
        cancel_current_tts()

        if self._active_llm_gen is not None:
            try:
                await self._active_llm_gen.aclose()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Error closing active LLM generator: {e}")
            self._active_llm_gen = None

        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._current_turn_task = None

    async def _interrupt_turn(self):
        """Handle user interrupt/barge-in: cancel turn and notify client."""
        target_turn_id = self._active_turn_id
        await self._cancel_current_turn()
        if target_turn_id and self._mark_terminal_sent(target_turn_id):
            await self._send(MSG_INTERRUPTED, {"turn_id": target_turn_id})

    async def _execute_turn(self, turn_id: int, audio_data: bytes):
        """STT → LLM → TTS pipeline execution for a single turn."""
        t0 = time.perf_counter()
        user_msg = None

        try:
            # 1. Speech-to-Text
            try:
                transcript = await transcribe_audio(audio_data, session_id=self.session_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"[{self.session_id}] STT failed in turn {turn_id}: {e}")
                await self._send_sanitized_error(turn_id)
                return

            if not transcript or not transcript.strip():
                logger.info(f"[{self.session_id}] Empty transcript in turn {turn_id}, ending turn.")
                if self._mark_terminal_sent(turn_id):
                    await self._send(MSG_TURN_END, {"turn_id": turn_id, "total_latency_ms": 0})
                return

            stt_latency = (time.perf_counter() - t0) * 1000
            logger.info(f"[{self.session_id}] ✅ STT done in turn {turn_id}: {stt_latency:.0f}ms — '{transcript[:60]}'")
            await self._send(MSG_TRANSCRIPT, {"text": transcript, "latency_ms": stt_latency, "turn_id": turn_id})

            # Record user message in conversation history
            user_msg = {"role": "user", "content": transcript}
            self.conversation_history.append(user_msg)

            # 2. LLM response streaming and sentence TTS
            try:
                await self._stream_and_speak(turn_id, t0)
            except LLMError as e:
                logger.warning(f"[{self.session_id}] LLMError in turn {turn_id}: {e}")
                # Prune orphan user message
                if user_msg is not None and self.conversation_history and self.conversation_history[-1] is user_msg:
                    self.conversation_history.pop()
                    logger.info(f"[{self.session_id}] Removed orphan user message from history.")

                # Speak fixed apology sentence
                await self._speak_apology(turn_id)
                return

        except asyncio.CancelledError:
            # Turn was cancelled; leave nothing in history
            if user_msg is not None and self.conversation_history and self.conversation_history[-1] is user_msg:
                self.conversation_history.pop()
            raise

        except Exception as e:
            logger.exception(f"[{self.session_id}] Unexpected error in turn {turn_id}: {e}")
            if user_msg is not None and self.conversation_history and self.conversation_history[-1] is user_msg:
                self.conversation_history.pop()
            await self._send_sanitized_error(turn_id)

        finally:
            # Only reset state to IDLE if this turn is still the active turn
            if turn_id == self._active_turn_id and self.state in (STATE_PROCESSING, STATE_SPEAKING):
                self.state = STATE_IDLE

    async def _stream_and_speak(self, turn_id: int, t0: float):
        """Stream LLM tokens → accumulate sentences → normalize → stream TTS."""
        full_response = []
        sentence_buffer = ""
        first_audio_sent = False
        llm_first_token_latency = None

        llm_gen = stream_llm_response(
            self.conversation_history[-10:],
            session_id=self.session_id
        )
        self._active_llm_gen = llm_gen

        try:
            async for token in llm_gen:
                if self._interrupt_event.is_set() or turn_id != self._active_turn_id:
                    logger.info(f"[{self.session_id}] Turn {turn_id} interrupted during LLM streaming")
                    break

                if llm_first_token_latency is None:
                    llm_first_token_latency = (time.perf_counter() - t0) * 1000
                    logger.info(f"[{self.session_id}] ✅ LLM first token in turn {turn_id}: {llm_first_token_latency:.0f}ms")
                    await self._send("latency", {"llm_first_token_ms": llm_first_token_latency, "turn_id": turn_id})

                await self._send(MSG_AI_TEXT, {"token": token, "turn_id": turn_id})
                full_response.append(token)
                sentence_buffer += token

                # Flush to TTS at sentence boundaries
                if _is_sentence_boundary(sentence_buffer):
                    sentence = " ".join(sentence_buffer.split())
                    sentence_buffer = ""
                    if sentence and not self._interrupt_event.is_set() and turn_id == self._active_turn_id:
                        sent_first = await self._send_sentence_audio(sentence, turn_id, first_audio_sent, t0)
                        if sent_first:
                            first_audio_sent = True

            # Flush any remaining text in sentence buffer
            if sentence_buffer and not self._interrupt_event.is_set() and turn_id == self._active_turn_id:
                sentence = " ".join(sentence_buffer.split())
                if sentence:
                    sent_first = await self._send_sentence_audio(sentence, turn_id, first_audio_sent, t0)
                    if sent_first:
                        first_audio_sent = True

            # Complete turn normally if still active
            if not self._interrupt_event.is_set() and turn_id == self._active_turn_id:
                if self._mark_terminal_sent(turn_id):
                    assistant_text = "".join(full_response)
                    if assistant_text:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": assistant_text
                        })
                    total_latency = (time.perf_counter() - t0) * 1000
                    logger.info(f"[{self.session_id}] ✅ Turn {turn_id} complete. Total: {total_latency:.0f}ms")
                    await self._send(MSG_TURN_END, {"total_latency_ms": total_latency, "turn_id": turn_id})
                    self.state = STATE_IDLE

        finally:
            try:
                await llm_gen.aclose()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Error closing LLM generator: {e}")
            if self._active_llm_gen is llm_gen:
                self._active_llm_gen = None

    async def _send_sentence_audio(self, sentence: str, turn_id: int, first_audio_sent: bool, t0: float) -> bool:
        """Stream TTS audio chunks for a single sentence. Returns True if first audio sent."""
        if self._interrupt_event.is_set() or turn_id != self._active_turn_id:
            return False

        t_tts = time.perf_counter()
        sent_any = False

        try:
            async for audio_chunk in stream_tts_audio(sentence, session_id=self.session_id):
                if self._interrupt_event.is_set() or turn_id != self._active_turn_id:
                    cancel_current_tts()
                    return sent_any

                if not first_audio_sent and not sent_any:
                    # Transition from PROCESSING to SPEAKING at exact moment first audio is sent
                    if turn_id == self._active_turn_id:
                        self.state = STATE_SPEAKING
                    await self._send(MSG_TURN_START, {"turn_id": turn_id})
                    first_audio_latency = (time.perf_counter() - t0) * 1000
                    logger.info(f"[{self.session_id}] ✅ First audio sent in turn {turn_id}: {first_audio_latency:.0f}ms")

                await self.websocket.send(audio_chunk)
                sent_any = True

            tts_duration = (time.perf_counter() - t_tts) * 1000
            logger.info(f"[{self.session_id}] 🔊 TTS sentence in turn {turn_id}: {tts_duration:.0f}ms — '{sentence[:40]}'")
            return sent_any

        except Exception as e:
            logger.error(f"[{self.session_id}] TTS error in turn {turn_id}: {e}")
            return sent_any

    async def _speak_apology(self, turn_id: int):
        """Speak fixed apology message on LLMError, ending turn cleanly."""
        logger.info(f"[{self.session_id}] Speaking apology line for turn {turn_id}")
        if self._interrupt_event.is_set() or turn_id != self._active_turn_id:
            return

        normalized = " ".join(APOLOGY_MESSAGE.split())
        await self._send(MSG_AI_TEXT, {"token": normalized, "turn_id": turn_id})

        # Transition to SPEAKING and synthesize audio
        if turn_id == self._active_turn_id:
            self.state = STATE_SPEAKING
        await self._send(MSG_TURN_START, {"turn_id": turn_id})

        try:
            async for audio_chunk in stream_tts_audio(normalized, session_id=self.session_id):
                if self._interrupt_event.is_set() or turn_id != self._active_turn_id:
                    cancel_current_tts()
                    break
                await self.websocket.send(audio_chunk)
        except Exception as e:
            logger.error(f"[{self.session_id}] Apology TTS error: {e}")

        if not self._interrupt_event.is_set() and turn_id == self._active_turn_id:
            if self._mark_terminal_sent(turn_id):
                await self._send(MSG_TURN_END, {"turn_id": turn_id, "total_latency_ms": 0})
            self.state = STATE_IDLE

    async def _send_sanitized_error(self, turn_id: Optional[int] = None):
        """Send sanitized error message to client without leaking server details."""
        effective_turn_id = turn_id if turn_id is not None else (self._active_turn_id or self.turn_id)
        if self._mark_terminal_sent(effective_turn_id):
            await self._send(INTERNAL_ERROR_PAYLOAD["type"], {
                "code": INTERNAL_ERROR_PAYLOAD["code"],
                "message": INTERNAL_ERROR_PAYLOAD["message"],
                "turn_id": effective_turn_id,
            })
        if effective_turn_id == self._active_turn_id:
            self.state = STATE_IDLE

    async def _send(self, msg_type: str, data: dict):
        """Send a JSON control message to the client."""
        try:
            payload = json.dumps({"type": msg_type, **data})
            await self.websocket.send(payload)
        except Exception as e:
            logger.debug(f"[{self.session_id}] Failed to send {msg_type}: {e}")

    async def cleanup(self):
        """Clean up all resources on disconnect."""
        logger.info(f"[{self.session_id}] Pipeline cleaning up...")
        await self._cancel_current_turn()
        self._speech_active = False
        self._audio_buffer.clear()
        self.state = STATE_IDLE
        logger.info(f"[{self.session_id}] Pipeline cleaned up cleanly")


def _is_sentence_boundary(text: str) -> bool:
    """
    Detect sentence boundaries for TTS chunking.
    Flushes to TTS at '.', '!', '?' or at ',' after long phrases.
    """
    text = text.strip()
    if not text:
        return False
    if text[-1] in ".!?":
        return True
    if text[-1] == "," and len(text) > 40:
        return True
    return False
