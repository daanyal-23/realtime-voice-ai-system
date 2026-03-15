"""
Voice Activity Detection (VAD)

Energy-based VAD with hysteresis to avoid triggering on noise.
No external ML model required — works on raw PCM energy levels.

Algorithm:
  - Track short-term RMS energy
  - Speech ON  when energy > SPEECH_THRESHOLD  for N consecutive frames
  - Speech OFF when energy < SILENCE_THRESHOLD for M consecutive frames (hysteresis)

Parameters tuned for 16kHz, 16-bit mono PCM with 20ms frames (320 samples).
"""

import collections
import logging

import numpy as np

logger = logging.getLogger("vad")

FRAME_SIZE = 320          # 20ms at 16kHz
SPEECH_THRESHOLD = 0.02   # RMS threshold to consider speech active
SILENCE_THRESHOLD = 0.01  # RMS threshold to consider silence

# Frames needed to confirm speech start / speech end
SPEECH_CONFIRM_FRAMES = 3
SILENCE_CONFIRM_FRAMES = 8   # 300ms of silence → end of utterance,was 15

# Background noise estimation window
NOISE_WINDOW = 50


class VAD:
    def __init__(self):
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speech = False
        self._sample_buffer = np.array([], dtype=np.float32)

        # Adaptive noise floor (rolling average of quiet frames)
        self._noise_history = collections.deque(maxlen=NOISE_WINDOW)
        self._adaptive_threshold = SPEECH_THRESHOLD

    def process(self, audio: np.ndarray) -> bool:
        """
        Process a chunk of normalized float32 audio [-1, 1].
        Returns True while speech is active, False during silence.
        Updates internal state.
        """
        # Accumulate samples and process in frames
        self._sample_buffer = np.concatenate([self._sample_buffer, audio])

        speech_detected_in_chunk = False

        while len(self._sample_buffer) >= FRAME_SIZE:
            frame = self._sample_buffer[:FRAME_SIZE]
            self._sample_buffer = self._sample_buffer[FRAME_SIZE:]
            frame_active = self._process_frame(frame)
            if frame_active:
                speech_detected_in_chunk = True

        return speech_detected_in_chunk or self._is_speech

    def _process_frame(self, frame: np.ndarray) -> bool:
        """Process a single 20ms frame."""
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # Update adaptive noise floor using quiet frames
        if rms < self._adaptive_threshold * 0.5:
            self._noise_history.append(rms)
            if len(self._noise_history) > 10:
                noise_floor = np.mean(self._noise_history)
                self._adaptive_threshold = max(
                    SPEECH_THRESHOLD,
                    noise_floor * 4.0   # threshold = 4x noise floor
                )

        if not self._is_speech:
            if rms > self._adaptive_threshold:
                self._speech_frame_count += 1
                self._silence_frame_count = 0
                if self._speech_frame_count >= SPEECH_CONFIRM_FRAMES:
                    self._is_speech = True
                    self._speech_frame_count = 0
                    logger.debug(f"VAD: speech ON (rms={rms:.4f}, threshold={self._adaptive_threshold:.4f})")
            else:
                self._speech_frame_count = 0
        else:
            if rms < SILENCE_THRESHOLD:
                self._silence_frame_count += 1
                self._speech_frame_count = 0
                if self._silence_frame_count >= SILENCE_CONFIRM_FRAMES:
                    self._is_speech = False
                    self._silence_frame_count = 0
                    logger.debug(f"VAD: speech OFF (rms={rms:.4f})")
            else:
                self._silence_frame_count = 0

        return self._is_speech

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    def reset(self):
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speech = False
        self._sample_buffer = np.array([], dtype=np.float32)
