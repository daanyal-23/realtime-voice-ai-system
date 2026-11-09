"""
Voice Activity Detection (VAD)

Energy-based VAD with adaptive hysteresis to avoid triggering on noise.
No external ML model required — works on raw PCM energy levels.

Algorithm:
  - Track short-term RMS energy per 20ms frame (320 samples at 16kHz)
  - Estimate background noise floor using an Exponential Moving Average (EMA) on non-speech frames
  - Adaptive hysteresis:
      speech_threshold  = max(SPEECH_THRESHOLD,  4.0 * noise_floor)
      silence_threshold = max(SILENCE_THRESHOLD, 2.0 * noise_floor)
  - Speech ON  when energy > speech_threshold  for N consecutive frames (default: 5 frames = 100ms)
  - Speech OFF when energy < silence_threshold for M consecutive frames (default: 15 frames = 300ms)

Parameters tuned for 16kHz, 16-bit mono PCM with 20ms frames (320 samples).
"""

import collections
import logging
from typing import Optional

import numpy as np

try:
    from config import (
        VAD_FRAME_SIZE,
        VAD_SPEECH_THRESHOLD,
        VAD_SILENCE_THRESHOLD,
        VAD_SPEECH_CONFIRM_FRAMES,
        VAD_SILENCE_CONFIRM_FRAMES,
        VAD_EMA_ALPHA,
    )
    FRAME_SIZE = VAD_FRAME_SIZE
    SPEECH_THRESHOLD = VAD_SPEECH_THRESHOLD
    SILENCE_THRESHOLD = VAD_SILENCE_THRESHOLD
    SPEECH_CONFIRM_FRAMES = VAD_SPEECH_CONFIRM_FRAMES
    SILENCE_CONFIRM_FRAMES = VAD_SILENCE_CONFIRM_FRAMES
    EMA_ALPHA = VAD_EMA_ALPHA
except Exception:
    FRAME_SIZE = 320          # 20ms at 16kHz
    SPEECH_THRESHOLD = 0.02   # Base RMS threshold to consider speech active
    SILENCE_THRESHOLD = 0.01  # Base RMS threshold to consider silence
    SPEECH_CONFIRM_FRAMES = 5 # 100ms at 20ms/frame
    SILENCE_CONFIRM_FRAMES = 15 # 300ms at 20ms/frame
    EMA_ALPHA = 0.05

logger = logging.getLogger("vad")

# Clamping bounds for estimated background noise floor
MIN_NOISE_FLOOR = 0.0001
MAX_NOISE_FLOOR = 0.05
NOISE_WINDOW = 50


class VAD:
    def __init__(
        self,
        speech_threshold: float = SPEECH_THRESHOLD,
        silence_threshold: float = SILENCE_THRESHOLD,
        speech_confirm_frames: int = SPEECH_CONFIRM_FRAMES,
        silence_confirm_frames: int = SILENCE_CONFIRM_FRAMES,
        ema_alpha: float = EMA_ALPHA,
    ):
        self.base_speech_threshold = speech_threshold
        self.base_silence_threshold = silence_threshold
        self.speech_confirm_frames = speech_confirm_frames
        self.silence_confirm_frames = silence_confirm_frames
        self.ema_alpha = ema_alpha

        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._active_speech_frames = 0
        self._is_speech = False
        self._sample_buffer = np.array([], dtype=np.float32)

        # Adaptive noise floor tracking (EMA of non-speech frames)
        self._noise_floor: Optional[float] = None
        self._noise_history = collections.deque(maxlen=NOISE_WINDOW)

    @property
    def _adaptive_threshold(self) -> float:
        """Dynamic speech threshold (backward-compatible attribute/property)."""
        if self._noise_floor is None:
            return self.base_speech_threshold
        return max(self.base_speech_threshold, self._noise_floor * 4.0)

    @_adaptive_threshold.setter
    def _adaptive_threshold(self, value: float):
        self._noise_floor = value / 4.0

    @property
    def current_speech_threshold(self) -> float:
        return self._adaptive_threshold

    @property
    def current_silence_threshold(self) -> float:
        """Dynamic silence threshold: max(SILENCE_THRESHOLD, 2.0 * noise_floor)."""
        if self._noise_floor is None:
            return self.base_silence_threshold
        return max(self.base_silence_threshold, self._noise_floor * 2.0)

    @property
    def noise_floor(self) -> float:
        return self._noise_floor if self._noise_floor is not None else MIN_NOISE_FLOOR

    def process(self, audio: np.ndarray) -> bool:
        """
        Process a chunk of normalized float32 audio [-1, 1].
        Returns True while speech is active, False during silence.
        Updates internal state.
        """
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
        curr_speech_thresh = self.current_speech_threshold

        if not self._is_speech:
            # Update adaptive noise floor using non-speech background frames
            if rms < curr_speech_thresh:
                clamped_rms = min(max(rms, MIN_NOISE_FLOOR), MAX_NOISE_FLOOR)
                if self._noise_floor is None:
                    self._noise_floor = clamped_rms
                else:
                    self._noise_floor = (1.0 - self.ema_alpha) * self._noise_floor + self.ema_alpha * clamped_rms
                self._noise_history.append(clamped_rms)

            if rms > curr_speech_thresh:
                self._speech_frame_count += 1
                self._silence_frame_count = 0
                if self._speech_frame_count >= self.speech_confirm_frames:
                    self._is_speech = True
                    self._active_speech_frames = self.speech_confirm_frames
                    self._speech_frame_count = 0
                    logger.debug(f"VAD: speech ON (rms={rms:.4f}, threshold={curr_speech_thresh:.4f})")
            else:
                self._speech_frame_count = 0
        else:
            curr_silence_thresh = self.current_silence_threshold
            if rms < curr_silence_thresh:
                self._silence_frame_count += 1
                self._speech_frame_count = 0
                if self._silence_frame_count >= self.silence_confirm_frames:
                    self._is_speech = False
                    self._silence_frame_count = 0
                    logger.debug(f"VAD: speech OFF (rms={rms:.4f}, silence_thresh={curr_silence_thresh:.4f})")
            else:
                self._silence_frame_count = 0
                self._active_speech_frames += 1

        return self._is_speech

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    @property
    def active_speech_frames(self) -> int:
        """Number of active speech frames in the current or most recent speech segment (excluding silence hangover)."""
        return self._active_speech_frames

    @property
    def active_speech_ms(self) -> float:
        """Active speech duration in milliseconds."""
        return (self._active_speech_frames * FRAME_SIZE / 16000.0) * 1000.0

    @property
    def active_speech_duration_s(self) -> float:
        """Active speech duration in seconds."""
        return (self._active_speech_frames * FRAME_SIZE) / 16000.0

    def reset(self):
        """Reset internal counters, buffers, and noise tracking."""
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._active_speech_frames = 0
        self._is_speech = False
        self._sample_buffer = np.array([], dtype=np.float32)
        self._noise_floor = None
        self._noise_history.clear()
