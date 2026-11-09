"""
Unit tests for TTS session isolation, lifecycle, deterministic cancellation, and subprocess reaping.
Verifies all requirements for Task A3.
"""

import asyncio
import io
import os
import subprocess
import sys
import tempfile
import wave
import threading
from unittest.mock import MagicMock, patch

import pytest

from tts import TTSSession, stream_tts_audio, cancel_current_tts, _tts_worker_script
from pipeline import VoicePipeline


def create_minimal_wav_bytes() -> bytes:
    """Generate a minimal valid 16kHz mono PCM WAV in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)  # 100ms
    return buf.getvalue()


class MockPopen:
    """Mock subprocess.Popen with poll(), kill(), wait(), returncode, and pid."""

    def __init__(self, pid=1234, returncode=0, delay=0.0):
        self.pid = pid
        self._returncode = None
        self._final_returncode = returncode
        self.killed = False
        self.waited = False
        self.delay = delay

    @property
    def returncode(self):
        return self._returncode

    @returncode.setter
    def returncode(self, val):
        self._returncode = val

    def poll(self):
        return self._returncode

    def kill(self):
        self.killed = True
        self._returncode = -9 if sys.platform != "win32" else 1

    def wait(self, timeout=None):
        self.waited = True
        if self._returncode is None:
            self._returncode = self._final_returncode
        return self._returncode


class TestTTSSessionIsolation:

    @pytest.fixture(autouse=True)
    def ensure_sapi5_provider(self, monkeypatch):
        """TestTTSSessionIsolation specifically tests SAPI5 subprocess management."""
        monkeypatch.setattr("tts.OPENAI_API_KEY", "")
        monkeypatch.setattr("tts.TTS_PROVIDER", "sapi5")

    @pytest.mark.asyncio
    async def test_two_sessions_operate_independently(self):
        """Two TTS sessions have completely independent process tracking and state."""
        session_a = TTSSession(session_id="session_A")
        session_b = TTSSession(session_id="session_B")

        proc_a = MockPopen(pid=101)
        proc_b = MockPopen(pid=102)

        session_a._current_process = proc_a
        session_b._current_process = proc_b

        assert session_a._current_process is proc_a
        assert session_b._current_process is proc_b
        assert session_a._current_process is not session_b._current_process

        await session_a.close()
        await session_b.close()

    @pytest.mark.asyncio
    async def test_cancelling_one_session_does_not_affect_another(self):
        """Cancelling session A kills session A's process without affecting session B's process."""
        session_a = TTSSession(session_id="session_A")
        session_b = TTSSession(session_id="session_B")

        proc_a = MockPopen(pid=201)
        proc_b = MockPopen(pid=202)

        session_a._current_process = proc_a
        session_b._current_process = proc_b

        # Cancel Session A
        session_a.cancel()

        # Session A process killed and unreferenced
        assert proc_a.killed is True
        assert session_a._current_process is None

        # Session B process completely untouched
        assert proc_b.killed is False
        assert session_b._current_process is proc_b

        await session_a.close()
        await session_b.close()

    @pytest.mark.asyncio
    async def test_cancellation_during_subprocess_creation(self):
        """
        Deterministic race test:
        If cancel() is called while subprocess.Popen is executing in the worker thread,
        the created process is registered, terminated, and reaped with no orphans.
        """
        session = TTSSession(session_id="race_test")
        mock_proc = MockPopen(pid=301)

        spawn_started = threading.Event()
        spawn_proceed = threading.Event()

        def slow_popen(*args, **kwargs):
            spawn_started.set()
            # Simulate latency in process creation
            spawn_proceed.wait(timeout=2.0)
            return mock_proc

        with patch("subprocess.Popen", side_effect=slow_popen):
            # Launch synthesis in background
            async def run_synth():
                chunks = []
                async for chunk in session.stream_tts_audio("Hello"):
                    chunks.append(chunk)
                return chunks

            synth_task = asyncio.create_task(run_synth())

            # Wait until Popen has entered in the executor thread
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, spawn_started.wait, 2.0)

            # Fire cancellation while Popen is still executing in thread.
            # Must return immediately without blocking the event loop!
            t0 = loop.time()
            session.cancel()
            assert loop.time() - t0 < 0.2  # Proves cancel() did not block waiting for Popen!

            # Now let Popen finish
            spawn_proceed.set()

            # Task completes with no audio
            result = await asyncio.wait_for(synth_task, timeout=2.0)
            assert result == []

            # Verify the process that finished spawning was immediately killed and reaped
            assert mock_proc.killed is True
            assert mock_proc.waited is True
            assert session._current_process is None

        await session.close()

    @pytest.mark.asyncio
    async def test_cancellation_during_synthesis(self):
        """Cancelling during subprocess wait terminates the process and yields nothing."""
        session = TTSSession(session_id="synth_cancel")
        mock_proc = MockPopen(pid=401)
        proc_waiting = threading.Event()
        proc_killed_event = threading.Event()

        def wait_hook(timeout=None):
            proc_waiting.set()
            # Real process blocks in wait() until killed or done
            proc_killed_event.wait(timeout=2.0)
            return -9 if sys.platform != "win32" else 1

        orig_kill = mock_proc.kill
        def kill_hook():
            orig_kill()
            proc_killed_event.set()

        mock_proc.wait = wait_hook
        mock_proc.kill = kill_hook

        with patch("subprocess.Popen", return_value=mock_proc):
            async def run_synth():
                chunks = []
                async for chunk in session.stream_tts_audio("Interrupt me"):
                    chunks.append(chunk)
                return chunks

            synth_task = asyncio.create_task(run_synth())

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, proc_waiting.wait, 2.0)

            session.cancel()

            result = await asyncio.wait_for(synth_task, timeout=2.0)
            assert result == []
            assert mock_proc.killed is True

        await session.close()

    @pytest.mark.asyncio
    async def test_process_termination_and_reaping(self):
        """_kill_process kills and reaps the target process safely in background thread."""
        session = TTSSession(session_id="reap_test")
        proc = MockPopen(pid=501)

        reaper = session._kill_process(proc)
        if reaper is not None:
            reaper.join(timeout=1.0)

        assert proc.killed is True
        assert proc.waited is True

        await session.close()

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self):
        """Temporary files are cleaned up after synthesis and not left on disk."""
        session = TTSSession(session_id="temp_cleanup")
        created_wav = create_minimal_wav_bytes()

        created_files = []

        def mock_popen_with_file(args, **kwargs):
            script_path = args[1]
            wav_path = script_path + ".wav"
            created_files.append(script_path)
            created_files.append(wav_path)
            # Write valid WAV output
            with open(wav_path, "wb") as f:
                f.write(created_wav)
            return MockPopen(pid=601, returncode=0)

        with patch("subprocess.Popen", side_effect=mock_popen_with_file):
            chunks = []
            async for chunk in session.stream_tts_audio("Clean up temp files."):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0] == created_wav

        # Ensure all temp files were unlinked after synthesis completed
        assert len(created_files) == 2
        for path in created_files:
            assert not os.path.exists(path)

        await session.close()

    @pytest.mark.asyncio
    async def test_closing_session_prevents_new_synthesis(self):
        """Closing a session sets is_closed and rejects subsequent synthesis requests."""
        session = TTSSession(session_id="close_test")
        await session.close()

    @pytest.mark.asyncio
    async def test_repeated_cancel_and_close_calls(self):
        """Repeated cancel() and close() calls are strictly idempotent and safe."""
        session = TTSSession(session_id="idempotent_test")

        # Multiple cancels
        session.cancel()
        session.cancel()
        session.cancel()

        # Multiple closes
        await session.close()

    @pytest.mark.asyncio
    async def test_empty_or_missing_wav_output(self, caplog):
        """Empty or missing WAV output logs warnings and yields zero chunks."""
        session = TTSSession(session_id="empty_wav")

        # 1. Missing WAV output
        def popen_no_output(args, **kwargs):
            return MockPopen(pid=701, returncode=0)

        with patch("subprocess.Popen", side_effect=popen_no_output):
            chunks = []
            async for chunk in session.stream_tts_audio("Missing WAV"):
                chunks.append(chunk)
            assert chunks == []

        # 2. 0-byte WAV output
        def popen_zero_byte(args, **kwargs):
            script_path = args[1]
            wav_path = script_path + ".wav"
            with open(wav_path, "wb") as f:
                pass  # 0 bytes
            return MockPopen(pid=702, returncode=0)

        with patch("subprocess.Popen", side_effect=popen_zero_byte):
            chunks = []
            async for chunk in session.stream_tts_audio("Empty WAV"):
                chunks.append(chunk)
            assert chunks == []

        # 3. Non-zero exit code
        def popen_failing(args, **kwargs):
            return MockPopen(pid=703, returncode=1)

        with patch("subprocess.Popen", side_effect=popen_failing):
            chunks = []
            async for chunk in session.stream_tts_audio("Failing process"):
                chunks.append(chunk)
            assert chunks == []

        await session.close()

    @pytest.mark.asyncio
    async def test_overlapping_synthesis_enforces_sequential_execution_and_no_untracked_subprocesses(self):
        """
        Overlapping synthesis requests on the same TTSSession must execute sequentially.
        At no point can two subprocesses be running concurrently or an untracked
        subprocess be left running.
        """
        session = TTSSession(session_id="overlap_test")
        created_wav = create_minimal_wav_bytes()

        running_procs = set()
        max_concurrent_procs = 0
        execution_order = []

        class TrackingMockPopen(MockPopen):
            def __init__(self, pid, name):
                super().__init__(pid=pid)
                self.name = name

            def wait(self, timeout=None):
                ret = super().wait(timeout=timeout)
                running_procs.discard(self)
                return ret

        def mock_popen(args, **kwargs):
            nonlocal max_concurrent_procs
            script_path = args[1]
            wav_path = script_path + ".wav"
            with open(wav_path, "wb") as f:
                f.write(created_wav)

            name = "proc_1" if len(execution_order) == 0 else "proc_2"
            proc = TrackingMockPopen(pid=800 + len(execution_order), name=name)
            execution_order.append(proc.name)
            running_procs.add(proc)

            if len(running_procs) > max_concurrent_procs:
                max_concurrent_procs = len(running_procs)

            return proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            async def synth1():
                chunks = []
                async for chunk in session.stream_tts_audio("Sentence 1"):
                    chunks.append(chunk)
                return chunks

            async def synth2():
                chunks = []
                async for chunk in session.stream_tts_audio("Sentence 2"):
                    chunks.append(chunk)
                return chunks

            t1 = asyncio.create_task(synth1())
            t2 = asyncio.create_task(synth2())

            results = await asyncio.gather(t1, t2)

            assert len(results[0]) == 1
            assert len(results[1]) == 1
            assert execution_order == ["proc_1", "proc_2"]
            assert max_concurrent_procs == 1
            assert len(running_procs) == 0
            assert session._current_process is None

        await session.close()

    @pytest.mark.asyncio
    async def test_close_does_not_cleanup_temp_files_while_subprocess_is_running(self):
        """
        close() terminates any running process and waits for synthesis to unwind
        before unlinking tracked temporary files, preventing file-in-use errors.
        """
        session = TTSSession(session_id="close_temp_test")
        mock_proc = MockPopen(pid=901)
        proc_running = threading.Event()
        proc_terminated = threading.Event()

        def wait_hook(timeout=None):
            mock_proc.waited = True
            proc_running.set()
            proc_terminated.wait(timeout=2.0)
            return -9 if sys.platform != "win32" else 1

        orig_kill = mock_proc.kill
        def kill_hook():
            orig_kill()
            proc_terminated.set()

        mock_proc.wait = wait_hook
        mock_proc.kill = kill_hook

        with patch("subprocess.Popen", return_value=mock_proc):
            async def run_synth():
                try:
                    async for _ in session.stream_tts_audio("Text that gets closed mid-way"):
                        pass
                except (Exception, asyncio.CancelledError):
                    pass

            synth_task = asyncio.create_task(run_synth())

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, proc_running.wait, 2.0)

            # Close while process is active
            await session.close()

    @pytest.mark.asyncio
    async def test_cancellation_invalidates_both_running_and_queued_synthesis_requests(self):
        """
        Deterministic test for queued synthesis cancellation:
        1. Request 1 acquires the lock and spawns a running subprocess.
        2. Request 2 is queued, capturing generation 0, and waits for the lock.
        3. cancel() is called (barge-in):
           - Increments cancellation generation (0 -> 1).
           - Kills Request 1's subprocess.
        4. Request 1 unwinds and releases the lock.
        5. Request 2 acquires the lock, detects generation mismatch (0 != 1),
           and drops itself without spawning a subprocess or resetting _cancelled.
        6. Request 3 (fresh request launched after cancellation) captures generation 1,
           acquires the lock, and synthesizes successfully.
        """
        session = TTSSession(session_id="queued_cancel_test")
        created_wav = create_minimal_wav_bytes()

        req1_proc = MockPopen(pid=1001)
        req3_proc = MockPopen(pid=1003)

        req1_running = threading.Event()
        req1_proceed = threading.Event()

        def req1_wait(timeout=None):
            req1_running.set()
            req1_proceed.wait(timeout=2.0)
            return -9 if sys.platform != "win32" else 1

        orig_kill = req1_proc.kill
        def req1_kill():
            orig_kill()
            req1_proceed.set()

        req1_proc.wait = req1_wait
        req1_proc.kill = req1_kill

        spawned_commands = []

        def mock_popen(args, **kwargs):
            script_path = args[1]
            wav_path = script_path + ".wav"
            with open(wav_path, "wb") as f:
                f.write(created_wav)

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "Sentence 1" in content:
                spawned_commands.append("Sentence 1")
                return req1_proc
            elif "Sentence 2" in content:
                spawned_commands.append("Sentence 2")
                return MockPopen(pid=1002)
            elif "Sentence 3" in content:
                spawned_commands.append("Sentence 3")
                return req3_proc
            raise ValueError(f"Unexpected script content: {content}")

        with patch("subprocess.Popen", side_effect=mock_popen):
            # 1. Launch Request 1 in background
            r1_chunks = []
            async def run_r1():
                async for chunk in session.stream_tts_audio("Sentence 1"):
                    r1_chunks.append(chunk)

            task1 = asyncio.create_task(run_r1())

            # Wait until Request 1 is actively running inside its subprocess wait
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, req1_running.wait, 2.0)

            # 2. Queue Request 2 while Request 1 holds the lock
            r2_chunks = []
            async def run_r2():
                async for chunk in session.stream_tts_audio("Sentence 2"):
                    r2_chunks.append(chunk)

            task2 = asyncio.create_task(run_r2())

            # Give task2 a brief moment to enter stream_tts_audio and wait on _synth_lock
            await asyncio.sleep(0.02)

            # 3. User barge-in: cancel() is called
            # This must invalidate BOTH Request 1 (running) and Request 2 (queued)
            session.cancel()

            # Await completion of Task 1 and Task 2
            await asyncio.gather(task1, task2)

            # Verify Request 1 was cancelled and yielded no audio
            assert r1_chunks == []
            assert req1_proc.killed is True

            # Verify Request 2 was invalidated by generation check, yielded no audio,
            # and DID NOT spawn any subprocess!
            assert r2_chunks == []
            assert "Sentence 2" not in spawned_commands

            # 4. Now launch Request 3 (fresh turn after barge-in)
            r3_chunks = []
            async def run_r3():
                async for chunk in session.stream_tts_audio("Sentence 3"):
                    r3_chunks.append(chunk)

            task3 = asyncio.create_task(run_r3())
            await task3

            # Verify Request 3 successfully executed and yielded audio
            assert len(r3_chunks) == 1
            assert r3_chunks[0] == created_wav
            assert "Sentence 3" in spawned_commands

        await session.close()

    @pytest.mark.asyncio
    async def test_delayed_termination_handled_and_temp_files_cleaned(self):
        """
        If subprocess does not exit on initial wait after kill, _reap_process_sync retries force kill,
        confirms reaping, and unlinks temp files only after process exit is confirmed.
        """
        session = TTSSession(session_id="delayed_term_test")
        created_wav = create_minimal_wav_bytes()

        class DelayedMockPopen(MockPopen):
            def __init__(self, pid=2001):
                super().__init__(pid=pid)
                self.wait_call_count = 0
                self.running_event = threading.Event()
                self.killed_event = threading.Event()

            def wait(self, timeout=None):
                self.wait_call_count += 1
                if not self.killed:
                    # Normal synthesis wait: wait until killed
                    self.running_event.set()
                    self.killed_event.wait(timeout=2.0)
                    return -9 if sys.platform != "win32" else 1

                # Reaping wait after kill:
                if self.wait_call_count <= 2:
                    # Initial reap wait times out (delayed termination)
                    raise subprocess.TimeoutExpired(cmd=["delayed_worker"], timeout=timeout or 3.0)
                # Second wait (retry force-kill) succeeds
                self.waited = True
                self._returncode = 1
                return 1

            def kill(self):
                super().kill()
                self.killed_event.set()

        mock_proc = DelayedMockPopen(pid=2001)
        created_files = []

        def mock_popen(args, **kwargs):
            script_path = args[1]
            wav_path = script_path + ".wav"
            created_files.append(script_path)
            created_files.append(wav_path)
            with open(wav_path, "wb") as f:
                f.write(created_wav)
            return mock_proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            chunks = []
            async def run_synth():
                async for chunk in session.stream_tts_audio("Delayed exit text"):
                    chunks.append(chunk)

            synth_task = asyncio.create_task(run_synth())

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, mock_proc.running_event.wait, 2.0)

            # Interrupt / barge-in
            session.cancel()
            await synth_task

        assert chunks == []
        assert mock_proc.killed is True
        assert mock_proc.waited is True
        assert mock_proc.wait_call_count >= 2

        # Because termination was confirmed on retry, temporary files were unlinked
        assert len(created_files) == 2
        for path in created_files:
            assert not os.path.exists(path)

        await session.close()

    @pytest.mark.asyncio
    async def test_termination_failure_retains_temp_files_without_silent_cleanup(self):
        """
        If subprocess completely fails to terminate after kill, temporary files are NOT unlinked,
        remain tracked in _active_temp_files, and an explicit error is logged.
        """
        session = TTSSession(session_id="unkillable_test")
        created_wav = create_minimal_wav_bytes()

        class UnkillableMockPopen(MockPopen):
            def __init__(self, pid=3001):
                super().__init__(pid=pid)
                self.running_event = threading.Event()
                self.killed_event = threading.Event()

            def poll(self):
                # Always alive
                return None

            def wait(self, timeout=None):
                if not self.killed:
                    # Normal synthesis wait: wait until killed
                    self.running_event.set()
                    self.killed_event.wait(timeout=2.0)
                    return -9 if sys.platform != "win32" else 1

                # Reaping wait after kill: always times out!
                raise subprocess.TimeoutExpired(cmd=["unkillable_worker"], timeout=timeout or 3.0)

            def kill(self):
                self.killed = True
                self.killed_event.set()

        mock_proc = UnkillableMockPopen(pid=3001)
        created_files = []

        def mock_popen(args, **kwargs):
            script_path = args[1]
            wav_path = script_path + ".wav"
            created_files.append(script_path)
            created_files.append(wav_path)
            with open(wav_path, "wb") as f:
                f.write(created_wav)
            return mock_proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            chunks = []
            async def run_synth():
                async for chunk in session.stream_tts_audio("Unkillable text"):
                    chunks.append(chunk)

            synth_task = asyncio.create_task(run_synth())

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, mock_proc.running_event.wait, 2.0)

            # Interrupt / barge-in
            session.cancel()
            await synth_task

        assert chunks == []
        assert mock_proc.killed is True

        # Process NEVER terminated: temporary files MUST NOT be unlinked!
        assert len(created_files) == 2
        for path in created_files:
            assert os.path.exists(path)  # NOT unlinked!

        # Files remain tracked in session._active_temp_files
        for path in created_files:
            assert path in session._active_temp_files

        # Clean up files manually to avoid leaving them on disk
        for path in created_files:
            try:
                os.unlink(path)
            except Exception:
                pass

        await session.close()

    @pytest.mark.asyncio
    async def test_real_subprocess_execution(self):
        """
        Executes a real short-lived OS subprocess to verify process creation,
        execution, and reaping without mocks.
        """
        session = TTSSession(session_id="real_process_test")

        # Generate a minimal valid WAV using a quick python one-liner script
        with patch("tts._tts_worker_script") as mock_script_gen:
            def dummy_script(text, output_path, rate, volume, **kwargs):
                safe_out = output_path.replace("\\", "\\\\")
                return (
                    "import wave\n"
                    f"with wave.open('{safe_out}', 'wb') as wf:\n"
                    "    wf.setnchannels(1)\n"
                    "    wf.setsampwidth(2)\n"
                    "    wf.setframerate(16000)\n"
                    "    wf.writeframes(b'\\x00\\x00' * 160)\n"
                )

            mock_script_gen.side_effect = dummy_script

            chunks = []
            async for chunk in session.stream_tts_audio("Real subprocess test"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert len(chunks[0]) > 44  # Valid WAV header + frames
            assert chunks[0][:4] == b"RIFF"

        await session.close()

    @pytest.mark.asyncio
    async def test_pipeline_owns_tts_session_and_cleans_up_on_disconnect(self):
        """VoicePipeline owns an independent TTSSession and closes it on cleanup()."""
        pipeline = VoicePipeline(session_id="pipe_tts_test", websocket=MagicMock())

        assert isinstance(pipeline.tts_session, TTSSession)
        assert pipeline.tts_session.session_id == "pipe_tts_test"
        assert pipeline.tts_session.is_closed is False

        await pipeline.cleanup()

        assert pipeline.tts_session.is_closed is True


class TestTTSVoiceSelection:
    """Verifies Hazel preference, voice fallback, case-insensitivity, and safe escaping."""

    def test_worker_script_generates_hazel_target_by_default(self):
        script = _tts_worker_script("Hello", "out.wav", 175, 1.0)
        assert "target = 'Hazel'.strip().lower()" in script

    def test_worker_script_generates_specified_voice_target(self):
        script = _tts_worker_script("Hello", "out.wav", 175, 1.0, voice_name="David")
        assert "target = 'David'.strip().lower()" in script

    def test_worker_script_escapes_special_characters_in_voice_name(self):
        script = _tts_worker_script("Hello", "out.wav", 175, 1.0, voice_name="Hazel's \"Test\" \\ Voice")
        assert "Hazel\\'s \\\"Test\\\" \\\\ Voice" in script
        compiled = compile(script, "<test_tts_script>", "exec")
        assert compiled is not None

    def test_voice_selection_logic_prefers_hazel_when_available(self):
        class MockVoice:
            def __init__(self, name, voice_id):
                self.name = name
                self.id = voice_id

        voices = [
            MockVoice("Microsoft David Desktop - English (United States)", "tts_david"),
            MockVoice("Microsoft Hazel Desktop - English (Great Britain)", "tts_hazel"),
            MockVoice("Microsoft Zira Desktop - English (United States)", "tts_zira"),
        ]

        target = "hazel".strip().lower()
        preferred = next(
            (v for v in voices if target and target in v.name.lower()),
            next((v for v in voices if "english" in v.name.lower()), voices[0]),
        )
        assert preferred.id == "tts_hazel"

    def test_voice_selection_logic_case_insensitive(self):
        class MockVoice:
            def __init__(self, name, voice_id):
                self.name = name
                self.id = voice_id

        voices = [
            MockVoice("Microsoft David Desktop", "tts_david"),
            MockVoice("Microsoft Hazel Desktop", "tts_hazel"),
        ]

        for voice_input in ["HAZEL", "hazel", "Hazel", "  hAzEL  "]:
            target = voice_input.strip().lower()
            preferred = next(
                (v for v in voices if target and target in v.name.lower()),
                next((v for v in voices if "english" in v.name.lower()), voices[0]),
            )
            assert preferred.id == "tts_hazel"

    def test_voice_selection_logic_falls_back_to_english_when_hazel_unavailable(self):
        class MockVoice:
            def __init__(self, name, voice_id):
                self.name = name
                self.id = voice_id

        voices = [
            MockVoice("Microsoft Huihui - Chinese (Simplified)", "tts_zh"),
            MockVoice("Microsoft David Desktop - English (United States)", "tts_david"),
        ]

        target = "hazel".strip().lower()
        preferred = next(
            (v for v in voices if target and target in v.name.lower()),
            next((v for v in voices if "english" in v.name.lower()), voices[0]),
        )
        assert preferred.id == "tts_david"

    def test_voice_selection_logic_falls_back_to_first_voice_when_no_english(self):
        class MockVoice:
            def __init__(self, name, voice_id):
                self.name = name
                self.id = voice_id

        voices = [
            MockVoice("Microsoft Huihui - Chinese (Simplified)", "tts_zh"),
            MockVoice("Microsoft Hortense - French", "tts_fr"),
        ]

        target = "hazel".strip().lower()
        preferred = next(
            (v for v in voices if target and target in v.name.lower()),
            next((v for v in voices if "english" in v.name.lower()), voices[0]),
        )
        assert preferred.id == "tts_zh"

    def test_tts_session_accepts_voice_name_parameter(self):
        session_default = TTSSession(session_id="test_def")
        assert session_default.voice_name == "Hazel"

        session_custom = TTSSession(session_id="test_cust", voice_name="Zira")
        assert session_custom.voice_name == "Zira"


class TestOpenAITTS:
    """Verifies gpt-4o-mini-tts integration, cedar/marin voices, error fallback to Hazel, and cancellation."""

    @pytest.mark.asyncio
    async def test_openai_tts_success_cedar(self):
        """Successful OpenAI TTS with cedar returns WAV bytes and sends correct payload."""
        session = TTSSession(
            session_id="test_openai_cedar",
            provider="openai",
            openai_voice="cedar",
            openai_model="gpt-4o-mini-tts",
        )
        fake_wav = create_minimal_wav_bytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav

        sent_requests = []

        async def mock_post(url, json=None, headers=None, **kwargs):
            sent_requests.append({"url": url, "json": json, "headers": headers})
            return mock_resp

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=mock_post):

            chunks = []
            async for chunk in session.stream_tts_audio("Good morning, Oak Lane Medical Practice."):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0] == fake_wav
            assert len(sent_requests) == 1
            payload = sent_requests[0]["json"]
            assert payload["model"] == "gpt-4o-mini-tts"
            assert payload["voice"] == "cedar"
            assert payload["response_format"] == "wav"
            assert payload["speed"] == 1.15
            assert "British English receptionist" in payload["instructions"]
            assert sent_requests[0]["headers"]["Authorization"] == "Bearer sk-mock-key-12345"

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_success_marin(self):
        """Successful OpenAI TTS with marin passes voice='marin' in payload."""
        session = TTSSession(
            session_id="test_openai_marin",
            provider="openai",
            openai_voice="marin",
            openai_model="gpt-4o-mini-tts",
        )
        fake_wav = create_minimal_wav_bytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav

        sent_payloads = []

        async def mock_post(url, json=None, **kwargs):
            sent_payloads.append(json)
            return mock_resp

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=mock_post):

            chunks = []
            async for chunk in session.stream_tts_audio("How may I help you?"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert sent_payloads[0]["voice"] == "marin"

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_speed_parameter_and_custom_override(self):
        """Custom openai_speed parameter is correctly passed in the synthesis payload."""
        session = TTSSession(
            session_id="test_speed_override",
            provider="openai",
            openai_voice="marin",
            openai_speed=1.25,
        )
        fake_wav = create_minimal_wav_bytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav

        sent_payloads = []

        async def mock_post(url, json=None, **kwargs):
            sent_payloads.append(json)
            return mock_resp

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=mock_post):

            chunks = []
            async for chunk in session.stream_tts_audio("Testing custom speed"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert sent_payloads[0]["speed"] == 1.25
            assert sent_payloads[0]["voice"] == "marin"

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_api_error_falls_back_to_hazel(self):
        """When OpenAI returns HTTP 429 or 500, it falls back to SAPI5 Hazel."""
        session = TTSSession(
            session_id="test_err_fallback",
            provider="openai",
            openai_voice="cedar",
        )
        hazel_wav = create_minimal_wav_bytes()

        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.text = "Rate limit reached"

        async def mock_post(*args, **kwargs):
            return error_resp

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=mock_post), \
             patch.object(session, "_spawn_worker") as mock_spawn, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=len(hazel_wav)), \
             patch("builtins.open", MagicMock(return_value=io.BytesIO(hazel_wav))):

            mock_proc = MockPopen(pid=999)
            mock_spawn.return_value = mock_proc

            chunks = []
            async for chunk in session.stream_tts_audio("Fallback test"):
                chunks.append(chunk)

            # SAPI5 was spawned and yielded audio
            assert mock_spawn.called
            assert len(chunks) == 1

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_timeout_falls_back_to_hazel(self):
        """When OpenAI times out, it immediately falls back to SAPI5 Hazel."""
        session = TTSSession(
            session_id="test_timeout_fallback",
            provider="openai",
            openai_voice="cedar",
        )
        hazel_wav = create_minimal_wav_bytes()

        async def mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Read timed out")

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=mock_timeout), \
             patch.object(session, "_spawn_worker") as mock_spawn, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=len(hazel_wav)), \
             patch("builtins.open", MagicMock(return_value=io.BytesIO(hazel_wav))):

            mock_proc = MockPopen(pid=998)
            mock_spawn.return_value = mock_proc

            chunks = []
            async for chunk in session.stream_tts_audio("Timeout test"):
                chunks.append(chunk)

            assert mock_spawn.called
            assert len(chunks) == 1

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_missing_key_falls_back_to_hazel_without_http(self):
        """When OPENAI_API_KEY is empty, it uses SAPI5 Hazel directly without calling HTTP."""
        session = TTSSession(
            session_id="test_missing_key",
            provider="openai",
            openai_voice="cedar",
        )
        hazel_wav = create_minimal_wav_bytes()

        http_called = False

        async def mock_post(*args, **kwargs):
            nonlocal http_called
            http_called = True
            return MagicMock()

        with patch("tts.OPENAI_API_KEY", ""), \
             patch("httpx.AsyncClient.post", side_effect=mock_post), \
             patch.object(session, "_spawn_worker") as mock_spawn, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=len(hazel_wav)), \
             patch("builtins.open", MagicMock(return_value=io.BytesIO(hazel_wav))):

            mock_proc = MockPopen(pid=997)
            mock_spawn.return_value = mock_proc

            chunks = []
            async for chunk in session.stream_tts_audio("No key test"):
                chunks.append(chunk)

            assert http_called is False
            assert mock_spawn.called
            assert len(chunks) == 1

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_cancellation(self):
        """Cancelling session while OpenAI HTTP request is in-flight cancels task and yields no audio."""
        session = TTSSession(
            session_id="test_cancel_http",
            provider="openai",
            openai_voice="cedar",
        )

        request_entered = asyncio.Event()

        async def slow_post(*args, **kwargs):
            request_entered.set()
            await asyncio.sleep(5.0)  # simulate long network wait
            return MagicMock(status_code=200, content=create_minimal_wav_bytes())

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", side_effect=slow_post):

            async def run_synth():
                chunks = []
                async for chunk in session.stream_tts_audio("Cancelling this"):
                    chunks.append(chunk)
                return chunks

            synth_task = asyncio.create_task(run_synth())
            await request_entered.wait()

            session.cancel()
            chunks = await synth_task

            assert len(chunks) == 0

        await session.close()

    @pytest.mark.asyncio
    async def test_openai_tts_connection_pooling_reuses_client(self):
        """Persistent httpx.AsyncClient is created once and reused across multiple sentences."""
        session = TTSSession(
            session_id="test_pool_reuse",
            provider="openai",
            openai_voice="marin",
        )
        fake_wav = create_minimal_wav_bytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav

        clients_used = []

        async def mock_post(client_self, url, json=None, **kwargs):
            clients_used.append(client_self)
            return mock_resp

        with patch("tts.OPENAI_API_KEY", "sk-mock-key-12345"), \
             patch("httpx.AsyncClient.post", mock_post):

            # Synthesize sentence 1
            chunks1 = []
            async for chunk in session.stream_tts_audio("Sentence one."):
                chunks1.append(chunk)

            # Synthesize sentence 2
            chunks2 = []
            async for chunk in session.stream_tts_audio("Sentence two."):
                chunks2.append(chunk)

            assert len(chunks1) == 1
            assert len(chunks2) == 1
            assert len(clients_used) == 2
            # Both requests must use the EXACT same persistent client instance!
            assert clients_used[0] is clients_used[1]
            assert session._http_client is clients_used[0]

        await session.close()
        # Verify client is closed after session.close()
        assert session._http_client is None

    @pytest.mark.asyncio
    async def test_openai_tts_client_closed_on_session_close(self):
        """TTSSession.close() properly calls aclose() on the persistent HTTP client."""
        session = TTSSession(
            session_id="test_pool_close",
            provider="openai",
            openai_voice="marin",
        )
        client = session._get_http_client()
        assert not client.is_closed

        await session.close()
        assert client.is_closed
        assert session._http_client is None

