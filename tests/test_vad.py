"""
Tests for Voice Activity Detection (VAD).
Fully deterministic tests using synthetic numpy signals.
"""

import numpy as np
import pytest

from vad import (
    VAD,
    FRAME_SIZE,
    SPEECH_THRESHOLD,
    SILENCE_THRESHOLD,
    SPEECH_CONFIRM_FRAMES,
    SILENCE_CONFIRM_FRAMES,
)


def generate_sine_frame(amplitude: float, frequency: float = 440.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a 20ms sine wave frame of float32 samples."""
    t = np.linspace(0, FRAME_SIZE / sample_rate, FRAME_SIZE, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_silence_frame(amplitude: float = 0.0) -> np.ndarray:
    """Generate a 20ms silent frame."""
    if amplitude == 0.0:
        return np.zeros(FRAME_SIZE, dtype=np.float32)
    return np.full(FRAME_SIZE, amplitude, dtype=np.float32)


class TestVAD:
    def test_silence_never_triggers_speech(self):
        """Near-zero amplitude / silence should never trigger is_speech=True."""
        vad = VAD()
        silence_frame = generate_silence_frame(amplitude=0.0001)

        # Feed 50 frames of silence
        for _ in range(50):
            result = vad.process(silence_frame)
            assert result is False
            assert vad.is_speech is False

    def test_speech_detection_confirmation_frames(self):
        """
        A sustained signal above SPEECH_THRESHOLD should trigger speech
        detection exactly after SPEECH_CONFIRM_FRAMES consecutive frames.
        """
        vad = VAD()
        # Amplitude 0.1 sine wave has RMS ~ 0.0707, well above SPEECH_THRESHOLD (0.02)
        speech_frame = generate_sine_frame(amplitude=0.1)

        # First SPEECH_CONFIRM_FRAMES - 1 frames: is_speech remains False
        for frame_idx in range(SPEECH_CONFIRM_FRAMES - 1):
            result = vad.process(speech_frame)
            assert result is False, f"Triggered early on frame {frame_idx + 1}"
            assert vad.is_speech is False

        # On the SPEECH_CONFIRM_FRAMES-th frame, speech triggers ON
        result = vad.process(speech_frame)
        assert result is True
        assert vad.is_speech is True

        # Subsequent speech frames stay active
        result = vad.process(speech_frame)
        assert result is True
        assert vad.is_speech is True

    def test_silence_confirmation_turns_speech_off(self):
        """
        After active speech, silence for SILENCE_CONFIRM_FRAMES
        should turn is_speech back to False.
        """
        vad = VAD()
        speech_frame = generate_sine_frame(amplitude=0.1)
        silence_frame = generate_silence_frame(amplitude=0.0)

        # 1. Trigger speech ON
        for _ in range(SPEECH_CONFIRM_FRAMES):
            vad.process(speech_frame)
        assert vad.is_speech is True

        # 2. Feed SILENCE_CONFIRM_FRAMES - 1 silence frames: should stay True
        for frame_idx in range(SILENCE_CONFIRM_FRAMES - 1):
            vad.process(silence_frame)
            assert vad.is_speech is True, f"Turned off early on silence frame {frame_idx + 1}"

        # 3. On the SILENCE_CONFIRM_FRAMES-th frame, is_speech becomes False
        result = vad.process(silence_frame)
        assert vad.is_speech is False
        assert result is False

    def test_isolated_noise_spike_does_not_trigger_speech(self):
        """A single frame spike should not trigger speech if below confirm count."""
        vad = VAD()
        speech_frame = generate_sine_frame(amplitude=0.1)
        silence_frame = generate_silence_frame(amplitude=0.0)

        # 1 loud frame followed by silence
        vad.process(speech_frame)
        assert vad.is_speech is False

        vad.process(silence_frame)
        assert vad.is_speech is False

        # Another single loud frame
        vad.process(speech_frame)
        assert vad.is_speech is False

    def test_adaptive_noise_floor(self):
        """
        Adaptive threshold increases when exposed to consistent quiet background noise,
        preventing weak signals from triggering speech.
        """
        vad = VAD()

        # Constant low background noise with RMS = 0.008 (below 0.5 * 0.02 = 0.01)
        noise_frame = generate_silence_frame(amplitude=0.008)

        # Feed > 10 quiet noise frames to calibrate adaptive threshold (noise_floor * 4 = 0.032)
        for _ in range(20):
            vad.process(noise_frame)

        # Threshold should have adapted to 0.008 * 4.0 = 0.032
        assert vad._adaptive_threshold > SPEECH_THRESHOLD
        assert pytest.approx(vad._adaptive_threshold, rel=1e-2) == 0.032

        # A weak signal with RMS ~ 0.025 (amplitude ~ 0.035 sine) is above default 0.02
        # but below the adapted threshold 0.032, so it should NOT trigger speech
        weak_speech_frame = generate_sine_frame(amplitude=0.035)
        for _ in range(SPEECH_CONFIRM_FRAMES + 2):
            vad.process(weak_speech_frame)
        assert vad.is_speech is False

        # A stronger signal with RMS ~ 0.0707 (amplitude 0.1) exceeds 0.032 and triggers speech
        strong_speech_frame = generate_sine_frame(amplitude=0.1)
        for _ in range(SPEECH_CONFIRM_FRAMES):
            vad.process(strong_speech_frame)
        assert vad.is_speech is True

    def test_reset(self):
        """Reset clears speech state, sample buffer, and counters."""
        vad = VAD()
        speech_frame = generate_sine_frame(amplitude=0.1)

        for _ in range(SPEECH_CONFIRM_FRAMES):
            vad.process(speech_frame)
        assert vad.is_speech is True

        vad.reset()
        assert vad.is_speech is False
        assert vad._speech_frame_count == 0
        assert vad._silence_frame_count == 0
        assert len(vad._sample_buffer) == 0
