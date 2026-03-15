"""
Speech-to-Text using Groq Whisper API (whisper-large-v3-turbo).

Optimizations for low latency:
  - Strip leading/trailing silence before sending to Whisper
  - In-memory WAV conversion (no disk I/O)
  - Sends only the active speech portion — reduces Whisper processing time
"""

import io
import logging
import wave

import numpy as np
from groq import AsyncGroq

logger = logging.getLogger("stt")
client = AsyncGroq()

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
PADDING_MS = 100           # keep 100ms before/after speech for natural boundaries


def _strip_silence(pcm_data: bytes) -> bytes:
    """
    Remove leading and trailing silence from PCM audio.
    Keeps PADDING_MS of audio around the speech region.
    This significantly reduces the audio duration sent to Whisper.
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    frame_size = int(SAMPLE_RATE * 0.02)   # 20ms frames
    padding_frames = int(PADDING_MS / 20)

    energies = []
    for i in range(0, len(samples) - frame_size, frame_size):
        frame = samples[i:i + frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2)))

    if not energies:
        return pcm_data

    # Find first and last frame above silence threshold
    speech_frames = [i for i, e in enumerate(energies) if e > SILENCE_THRESHOLD]

    if not speech_frames:
        return pcm_data

    start_frame = max(0, speech_frames[0] - padding_frames)
    end_frame = min(len(energies), speech_frames[-1] + padding_frames + 1)

    start_sample = start_frame * frame_size
    end_sample = end_frame * frame_size

    trimmed = samples[start_sample:end_sample]
    trimmed_int16 = (trimmed * 32768).astype(np.int16)

    original_dur = len(samples) / SAMPLE_RATE
    trimmed_dur = len(trimmed_int16) / SAMPLE_RATE
    logger.debug(f"Silence strip: {original_dur:.2f}s → {trimmed_dur:.2f}s")

    return trimmed_int16.tobytes()


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert raw PCM bytes to WAV format in memory (no disk I/O)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def transcribe_audio(pcm_data: bytes, session_id: str = "") -> str:
    """
    Transcribe PCM audio bytes to text using Groq Whisper.
    Strips silence first to minimize audio duration sent to API.
    """
    # Strip silence to reduce audio size → faster Whisper response
    trimmed_pcm = _strip_silence(pcm_data)
    wav_data = _pcm_to_wav(trimmed_pcm)

    audio_file = io.BytesIO(wav_data)
    audio_file.name = "audio.wav"

    try:
        response = await client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_file,
            response_format="text",
            language="en",
        )
        transcript = response.strip() if isinstance(response, str) else response.text.strip()
        logger.debug(f"[{session_id}] Transcript: '{transcript[:80]}'")
        return transcript
    except Exception as e:
        logger.error(f"[{session_id}] Groq STT error: {e}")
        raise