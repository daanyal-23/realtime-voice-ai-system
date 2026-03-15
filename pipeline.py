"""
VoicePipeline: Orchestrates the full STT → LLM → TTS loop.

Design:
  - Audio chunks arrive over WebSocket
  - A VAD (energy-based) detects speech start/end
  - On end-of-speech, audio is sent to STT (Whisper via OpenAI API)
  - Transcript goes to LLM (GPT-4o streaming)
  - LLM tokens are streamed to TTS (OpenAI TTS streaming)
  - TTS audio chunks are sent back over WebSocket
  - Interruption: user speaking while AI is talking cancels current TTS stream
"""

import asyncio
import json
import logging
import time
from typing import Optional

import numpy as np

from stt import transcribe_audio
from llm import stream_llm_response
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


class VoicePipeline:
    def __init__(self, session_id: str, websocket):
        self.session_id = session_id
        self.websocket = websocket
        self.vad = VAD()

        # Conversation history for LLM context
        self.conversation_history = []

        # Interruption control
        self._ai_speaking = False
        self._interrupt_event = asyncio.Event()
        self._current_tts_task: Optional[asyncio.Task] = None

        # Audio buffer for current speech segment
        self._audio_buffer = bytearray()
        self._speech_active = False

        logger.info(f"[{self.session_id}] Pipeline initialized")

    async def run(self, websocket):
        """Main loop: receive audio/control messages from client."""
        await self._send(MSG_STATUS, {"message": "connected", "session_id": self.session_id})

        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Raw PCM audio data
                    await self._handle_audio_chunk(message)
                elif isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_control(data)
            except json.JSONDecodeError:
                logger.warning(f"[{self.session_id}] Invalid JSON control message")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error handling message: {e}", exc_info=True)
                await self._send(MSG_ERROR, {"message": str(e)})

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
            logger.info(f"[{self.session_id}] Speech start detected")

            # If AI is talking, interrupt it
            if self._ai_speaking:
                await self._interrupt_ai()

        if self._speech_active:
            self._audio_buffer.extend(chunk)

        MAX_AUDIO_BYTES = 16000 * 2 * 8  # 8 seconds max (16kHz, 16-bit)

        if (not speech_detected and self._speech_active) or \
   (self._speech_active and len(self._audio_buffer) > MAX_AUDIO_BYTES):
            # Speech just ended — check minimum duration (300ms at 16kHz 16-bit = 9600 bytes)
            if len(self._audio_buffer) > 9600:
                self._speech_active = False
                audio_data = bytes(self._audio_buffer)
                self._audio_buffer = bytearray()
                logger.info(f"[{self.session_id}] Speech end detected, {len(audio_data)} bytes")
                # Process asynchronously so we don't block audio ingestion
                asyncio.create_task(self._process_speech(audio_data))
            else:
                # Too short, likely noise
                self._speech_active = False
                self._audio_buffer = bytearray()

    async def _handle_control(self, data: dict):
        """Handle JSON control messages from client."""
        msg_type = data.get("type")

        if msg_type == "interrupt":
            if self._ai_speaking:
                await self._interrupt_ai()

        elif msg_type == "reset":
            self.conversation_history = []
            await self._send(MSG_STATUS, {"message": "conversation reset"})

        elif msg_type == "ping":
            await self._send("pong", {"ts": time.time()})

    async def _interrupt_ai(self):
        """Cancel ongoing TTS/LLM generation."""
        logger.info(f"[{self.session_id}] ⚡ INTERRUPT CALLED")
        self._interrupt_event.set()
        logger.info(f"[{self.session_id}] ⚡ Calling cancel_current_tts()")
        cancel_current_tts()
        logger.info(f"[{self.session_id}] ⚡ cancel_current_tts() done")
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
        self._ai_speaking = False
        await self._send(MSG_INTERRUPTED, {})

    async def _process_speech(self, audio_data: bytes):
        """STT → LLM → TTS pipeline for a complete speech segment."""
        t0 = time.perf_counter()

        # 1. Speech-to-Text
        try:
            transcript = await transcribe_audio(audio_data, session_id=self.session_id)
        except Exception as e:
            logger.error(f"[{self.session_id}] STT failed: {e}")
            await self._send(MSG_ERROR, {"message": "Speech recognition failed"})
            return

        if not transcript or not transcript.strip():
            logger.info(f"[{self.session_id}] Empty transcript, skipping")
            return

        stt_latency = (time.perf_counter() - t0) * 1000
        logger.info(f"[{self.session_id}] ✅ STT done: {stt_latency:.0f}ms — '{transcript[:60]}'")
        await self._send(MSG_TRANSCRIPT, {"text": transcript, "latency_ms": stt_latency})

        self.conversation_history.append({"role": "user", "content": transcript})

        # 2. RAG context retrieval , time it separately
        t_rag = time.perf_counter()
        self._interrupt_event.clear()
        self._ai_speaking = True
        await self._send(MSG_TURN_START, {})

        logger.info(f"[{self.session_id}] ⏳ Starting LLM stream...")
        self._current_tts_task = asyncio.create_task(
            self._stream_response(t0)
        )
        try:
            await self._current_tts_task
        except asyncio.CancelledError:
            logger.info(f"[{self.session_id}] TTS task cancelled")

    async def _stream_response(self, t0: float):
        """
        Stream LLM tokens → accumulate into sentences → stream TTS per sentence.
        This minimizes first-audio latency: we don't wait for the full LLM response.
        """
        full_response = []
        sentence_buffer = ""
        first_audio_sent = False
        llm_first_token_latency = None

        try:
            logger.info(f"[{self.session_id}] ⏳ LLM stream starting...")

            async for token in stream_llm_response(
                self.conversation_history[-10:],   # limit history to avoid slow responses
                session_id=self.session_id
            ):
                if self._interrupt_event.is_set():
                    break

                if llm_first_token_latency is None:
                    llm_first_token_latency = (time.perf_counter() - t0) * 1000
                    logger.info(f"[{self.session_id}] ✅ LLM first token: {llm_first_token_latency:.0f}ms")
                    await self._send("latency", {"llm_first_token_ms": llm_first_token_latency})

                await self._send(MSG_AI_TEXT, {"token": token})
                full_response.append(token)
                sentence_buffer += token

                # Flush to TTS at sentence boundaries for low latency
                if _is_sentence_boundary(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    if sentence and not self._interrupt_event.is_set():
                        if not first_audio_sent:
                            pre_tts_latency = (time.perf_counter() - t0) * 1000
                            logger.info(f"[{self.session_id}] ⏳ First sentence to TTS: {pre_tts_latency:.0f}ms — '{sentence[:50]}'")

                        t_tts = time.perf_counter()
                        await self._send_tts_for_sentence(sentence)
                        tts_duration = (time.perf_counter() - t_tts) * 1000

                        if not first_audio_sent:
                            first_audio_latency = (time.perf_counter() - t0) * 1000
                            logger.info(f"[{self.session_id}] ✅ First audio sent: {first_audio_latency:.0f}ms (TTS took {tts_duration:.0f}ms)")
                            first_audio_sent = True
                        else:
                            logger.info(f"[{self.session_id}] 🔊 TTS sentence: {tts_duration:.0f}ms")

            # Flush any remaining text
            if sentence_buffer.strip() and not self._interrupt_event.is_set():
                t_tts = time.perf_counter()
                await self._send_tts_for_sentence(sentence_buffer.strip())
                tts_duration = (time.perf_counter() - t_tts) * 1000
                logger.info(f"[{self.session_id}] 🔊 TTS final chunk: {tts_duration:.0f}ms")

            if not self._interrupt_event.is_set():
                assistant_text = "".join(full_response)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_text
                })
                total_latency = (time.perf_counter() - t0) * 1000
                logger.info(f"[{self.session_id}] ✅ Turn complete. Total: {total_latency:.0f}ms")
                await self._send(MSG_TURN_END, {"total_latency_ms": total_latency})

        finally:
            self._ai_speaking = False

    async def _send_tts_for_sentence(self, sentence: str):
        """Stream TTS audio chunks for a single sentence."""
        # Check interrupt BEFORE starting synthesis , skip sentence entirely
        if self._interrupt_event.is_set():
            return
        try:
            async for audio_chunk in stream_tts_audio(sentence, session_id=self.session_id):
                if self._interrupt_event.is_set():
                    return
                await self.websocket.send(audio_chunk)
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS error: {e}")

    async def _send_tts_for_sentence(self, sentence: str):
        """Stream TTS audio chunks for a single sentence."""
        # Check interrupt before even starting synthesis
        if self._interrupt_event.is_set():
            logger.info(f"[{self.session_id}] Skipping TTS — interrupted")
            return
        try:
            async for audio_chunk in stream_tts_audio(sentence, session_id=self.session_id):
                if self._interrupt_event.is_set():
                    cancel_current_tts()
                    return
                await self.websocket.send(audio_chunk)
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS error: {e}")
            
    async def _send(self, msg_type: str, data: dict):
        """Send a JSON control message to the client."""
        try:
            payload = json.dumps({"type": msg_type, **data})
            await self.websocket.send(payload)
        except Exception as e:
            logger.debug(f"[{self.session_id}] Failed to send {msg_type}: {e}")

    async def cleanup(self):
        """Clean up resources on disconnect."""
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
        logger.info(f"[{self.session_id}] Pipeline cleaned up")


def _is_sentence_boundary(text: str) -> bool:
    """
    Detect sentence boundaries for TTS chunking.
    We flush to TTS at '.', '!', '?' followed by space or end,
    or at ',' for longer phrases to reduce first-audio latency.
    """
    text = text.strip()
    if not text:
        return False
    # Hard boundaries
    if text[-1] in ".!?":
        return True
    # Soft boundary: comma after sufficient text (reduces latency on long sentences)
    if text[-1] == "," and len(text) > 40:
        return True
    return False
