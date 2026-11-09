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

    def test_80ms_impulse_noise_rejected(self):
        """
        An 80ms (4 frames) loud impulse noise (e.g. firecracker, click, or door slam)
        must NOT trigger speech detection when SPEECH_CONFIRM_FRAMES=5 (100ms).
        """
        vad = VAD()
        # Loud impulse frame with high amplitude 0.25 (RMS ~ 0.177)
        impulse_frame = generate_sine_frame(amplitude=0.25, frequency=800.0)
        silence_frame = generate_silence_frame(amplitude=0.0)

        # 4 consecutive loud frames = 80ms
        for frame_idx in range(4):
            result = vad.process(impulse_frame)
            assert result is False, f"Impulse triggered early on frame {frame_idx + 1}"
            assert vad.is_speech is False

        # Followed by quiet/silence
        for _ in range(10):
            result = vad.process(silence_frame)
            assert result is False
            assert vad.is_speech is False

    def test_longer_environmental_noise_bursts(self):
        """
        Longer environmental noise below speech threshold (e.g. fan hum at RMS 0.015)
        adapts the noise floor and does not trigger speech. Transient bursts < 100ms are ignored.
        """
        vad = VAD()
        # Ambient noise with RMS ~ 0.014
        noise_frame = generate_silence_frame(amplitude=0.014)

        # Feed 30 frames (600ms) of ambient noise
        for _ in range(30):
            result = vad.process(noise_frame)
            assert result is False
            assert vad.is_speech is False

        # Threshold adapts to 0.014 * 4.0 = 0.056
        assert vad._adaptive_threshold >= 0.05

        # Transient 4-frame (80ms) burst of louder noise (amplitude 0.08)
        burst_frame = generate_sine_frame(amplitude=0.08)
        for _ in range(4):
            result = vad.process(burst_frame)
            assert result is False
            assert vad.is_speech is False

        # Return to ambient noise
        for _ in range(10):
            assert vad.process(noise_frame) is False
            assert vad.is_speech is False

    def test_normal_speech_onset(self):
        """
        Normal conversational speech (RMS ~ 0.0707) reliably triggers speech detection
        at exactly 100ms (frame 5) and stays active.
        """
        vad = VAD()
        speech_frame = generate_sine_frame(amplitude=0.1)

        # Frames 1..4: speech confirmation counting, is_speech is False
        for frame_idx in range(1, 5):
            result = vad.process(speech_frame)
            assert result is False, f"Speech triggered before 100ms on frame {frame_idx}"
            assert vad.is_speech is False

        # Frame 5 (100ms): speech triggers ON
        result = vad.process(speech_frame)
        assert result is True
        assert vad.is_speech is True

        # Subsequent speech frames stay active
        for _ in range(10):
            assert vad.process(speech_frame) is True
            assert vad.is_speech is True

    def test_200ms_pause_within_speech(self):
        """
        A 200ms (10 frames) pause between spoken words must NOT trigger speech end
        when SILENCE_CONFIRM_FRAMES=15 (300ms). Speech must resume seamlessly.
        """
        vad = VAD()
        speech_frame = generate_sine_frame(amplitude=0.1)
        silence_frame = generate_silence_frame(amplitude=0.0)

        # 1. Trigger speech ON (5 frames = 100ms)
        for _ in range(5):
            vad.process(speech_frame)
        assert vad.is_speech is True

        # 2. Inter-word pause: 10 frames (200ms) of silence
        for pause_frame in range(1, 11):
            vad.process(silence_frame)
            assert vad.is_speech is True, f"Premature speech end during 200ms pause at frame {pause_frame}"

        # 3. Speech resumes (5 more frames)
        for _ in range(5):
            vad.process(speech_frame)
            assert vad.is_speech is True

        # 4. Final utterance end: 15 frames (300ms) of silence
        for silence_idx in range(1, 15):
            vad.process(silence_frame)
            assert vad.is_speech is True, f"Speech ended early at silence frame {silence_idx}"

        # 15th silence frame confirms speech end
        result = vad.process(silence_frame)
        assert result is False
        assert vad.is_speech is False

    def test_speech_following_background_noise(self):
        """
        In presence of elevated background noise (RMS 0.012), VAD adapts:
        speech turns ON during speech, and cleanly turns OFF back into the noise floor
        without hanging indefinitely.
        """
        vad = VAD()
        bg_noise = generate_silence_frame(amplitude=0.012)
        speech_frame = generate_sine_frame(amplitude=0.12)

        # 1. Establish background noise for 25 frames
        for _ in range(25):
            vad.process(bg_noise)
        assert vad.is_speech is False

        # Silence threshold must be elevated to 0.012 * 2.0 = 0.024
        assert vad.current_silence_threshold >= 0.024

        # 2. User speaks above background noise (5 frames = 100ms)
        for _ in range(5):
            vad.process(speech_frame)
        assert vad.is_speech is True

        # 3. User finishes speaking; audio returns to background noise (amplitude 0.012 < 0.024)
        for silence_idx in range(1, 15):
            vad.process(bg_noise)
            assert vad.is_speech is True, f"Turned off early on frame {silence_idx}"

        # On the 15th frame, speech turns OFF cleanly despite ongoing background noise
        result = vad.process(bg_noise)
        assert result is False
        assert vad.is_speech is False

    def test_noise_floor_sweep(self):
        """
        Verify adaptive hysteresis across a sweep of ambient noise levels.
        At each level, speech threshold = max(0.02, 4*floor) and
        silence threshold = max(0.01, 2*floor).
        """
        for noise_amp in [0.002, 0.006, 0.012, 0.020]:
            vad = VAD()
            noise_frame = generate_silence_frame(amplitude=noise_amp)

            # Feed 20 noise frames
            for _ in range(20):
                vad.process(noise_frame)

            expected_speech_thresh = max(SPEECH_THRESHOLD, noise_amp * 4.0)
            expected_silence_thresh = max(SILENCE_THRESHOLD, noise_amp * 2.0)

            assert pytest.approx(vad.current_speech_threshold, rel=1e-2) == expected_speech_thresh
            assert pytest.approx(vad.current_silence_threshold, rel=1e-2) == expected_silence_thresh
            assert vad.is_speech is False
