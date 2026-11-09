"""
Text-to-Speech using pyttsx3 via isolated session subprocesses.

Session Isolation Design:
  - Each VoicePipeline owns an independent TTSSession instance.
  - Subprocesses are tracked strictly per-session; one session cannot affect another.
  - On interrupt: session.cancel() immediately terminates only its own active subprocess.
  - Cancellation during subprocess creation is handled deterministically via a threading lock
    and post-spawn check, guaranteeing zero orphaned processes.
  - Temporary files are unlinked only after the subprocess has fully exited and been reaped.
  - Subprocess completion/waiting runs in executor threads without blocking the event loop.
  - Closed sessions cleanly reject subsequent synthesis calls.
"""

import asyncio
import logging
import os
import sys
import subprocess
import tempfile
import threading
from typing import AsyncGenerator, Optional, Set

import httpx

logger = logging.getLogger("tts")

RATE = 175      # words per minute
VOLUME = 1.0    # 0.0 to 1.0

try:
    from config import (
        TTS_PROVIDER,
        TTS_VOICE,
        OPENAI_API_KEY,
        OPENAI_TTS_MODEL,
        OPENAI_TTS_VOICE,
        OPENAI_TTS_SPEED,
        OPENAI_TTS_INSTRUCTIONS,
    )
except Exception:
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "sapi5")
    TTS_VOICE = os.getenv("TTS_VOICE", "Hazel")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "cedar")
    OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.15"))
    OPENAI_TTS_INSTRUCTIONS = os.getenv(
        "OPENAI_TTS_INSTRUCTIONS",
        "Speak in a warm, professional British English receptionist voice. Use a natural, confident, moderately brisk conversational pace. Avoid unnecessarily long pauses. Keep pronunciation clear and reassuring.",
    )


def _tts_worker_script(
    text: str,
    output_path: str,
    rate: int,
    volume: float,
    voice_name: str = "Hazel",
) -> str:
    """
    Generate a self-contained Python script string that synthesizes
    text to a WAV file using pyttsx3.
    Runs as a subprocess so it can be killed instantly.
    Prefers the configured voice (default: Hazel), falling back to English or first voice.
    """
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    safe_path = output_path.replace("\\", "\\\\")
    safe_voice = (voice_name or "Hazel").replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    return f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', {rate})
engine.setProperty('volume', {volume})
voices = engine.getProperty('voices')
if voices:
    target = '{safe_voice}'.strip().lower()
    preferred = next(
        (v for v in voices if target and target in v.name.lower()),
        next((v for v in voices if 'english' in v.name.lower()), voices[0])
    )
    engine.setProperty('voice', preferred.id)
engine.save_to_file('{safe_text}', '{safe_path}')
engine.runAndWait()
engine.stop()
"""


class TTSSession:
    """
    Session-isolated Text-to-Speech manager.
    Guarantees strict isolation, killable subprocesses, generation-based cancellation
    of running and queued synthesis requests, and confirmed temporary-file reaping.
    """

    def __init__(
        self,
        session_id: str = "",
        voice_name: Optional[str] = None,
        provider: Optional[str] = None,
        openai_voice: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_instructions: Optional[str] = None,
        openai_speed: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.session_id = session_id
        self.voice_name = voice_name if voice_name is not None else TTS_VOICE
        self.provider = provider if provider is not None else TTS_PROVIDER
        self.openai_voice = openai_voice if openai_voice is not None else OPENAI_TTS_VOICE
        self.openai_model = openai_model if openai_model is not None else OPENAI_TTS_MODEL
        self.openai_instructions = openai_instructions if openai_instructions is not None else OPENAI_TTS_INSTRUCTIONS
        self.openai_speed = openai_speed if openai_speed is not None else OPENAI_TTS_SPEED
        self._http_client: Optional[httpx.AsyncClient] = http_client
        self._shared_client: bool = (http_client is not None)
        self._current_process: Optional[subprocess.Popen] = None
        self._active_http_tasks: Set[asyncio.Task] = set()
        self._committed_provider: Optional[str] = None
        self._lock = threading.Lock()
        self._synth_lock: Optional[asyncio.Lock] = None
        self._cancel_generation = 0
        self._cancelled = False
        self._closed = False
        self._active_temp_files: Set[str] = set()

    def _get_synth_lock(self) -> asyncio.Lock:
        if self._synth_lock is None:
            self._synth_lock = asyncio.Lock()
        return self._synth_lock

    def _get_http_client(self) -> httpx.AsyncClient:
        """
        Get or initialize the persistent httpx.AsyncClient for connection pooling.
        Reuses TCP and TLS connections across sentences to eliminate handshake latency.
        """
        with self._lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
                self._http_client = httpx.AsyncClient(timeout=timeout)
                self._shared_client = False
            return self._http_client

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def cancel(self) -> None:
        """
        Immediately kill active synthesis (process or HTTP request) for this session only,
        and invalidate all currently running and queued synthesis requests via generation bump.
        Thread-safe, non-blocking to event loop, and idempotent.
        """
        proc_to_kill = None
        http_tasks_to_cancel = []
        with self._lock:
            self._cancel_generation += 1
            self._cancelled = True
            self._committed_provider = None
            if self._current_process and self._current_process.poll() is None:
                proc_to_kill = self._current_process
                self._current_process = None
            http_tasks_to_cancel = list(self._active_http_tasks)
            self._active_http_tasks.clear()

        if proc_to_kill is not None:
            self._kill_process(proc_to_kill)
        for task in http_tasks_to_cancel:
            if not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass

    async def _synthesize_openai(
        self,
        text: str,
        req_generation: int,
        session_id: str,
    ) -> Optional[bytes]:
        """
        Synthesize text using OpenAI gpt-4o-mini-tts via httpx.
        Returns raw WAV bytes on success, or None on failure/missing key (prompting Hazel fallback).
        Never logs or reveals API keys.
        """
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.debug(f"[{session_id}] OPENAI_API_KEY not configured; using SAPI5 Hazel fallback.")
            return None

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "voice": self.openai_voice,
            "input": text,
            "instructions": self.openai_instructions,
            "response_format": "wav",
            "speed": self.openai_speed,
        }

        async def _do_post_with_retry():
            client = self._get_http_client()
            try:
                return await client.post(url, json=payload, headers=headers)
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.WriteError, httpx.PoolTimeout) as e:
                logger.warning(
                    f"[{session_id}] Pooled connection error ({type(e).__name__}); retrying on fresh connection"
                )
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0)) as retry_client:
                    return await retry_client.post(url, json=payload, headers=headers)

        loop = asyncio.get_running_loop()
        task = loop.create_task(_do_post_with_retry())
        with self._lock:
            if self._cancelled or self._closed or req_generation != self._cancel_generation:
                task.cancel()
                return None
            self._active_http_tasks.add(task)

        try:
            resp = await task
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] OpenAI TTS task cancelled for gen {req_generation}")
            return None
        except Exception as e:
            logger.warning(
                f"[{session_id}] OpenAI TTS request exception ({type(e).__name__}: {e})"
            )
            return None
        finally:
            with self._lock:
                self._active_http_tasks.discard(task)

        with self._lock:
            if self._cancelled or self._closed or req_generation != self._cancel_generation:
                return None

        if resp.status_code != 200:
            logger.warning(
                f"[{session_id}] OpenAI TTS returned HTTP {resp.status_code}: {resp.text[:120]}"
            )
            return None

        wav_bytes = resp.content
        if not wav_bytes or len(wav_bytes) < 44:
            logger.warning(
                f"[{session_id}] OpenAI TTS returned invalid WAV ({len(wav_bytes) if wav_bytes else 0} bytes)"
            )
            return None

        return wav_bytes

    def _reap_process_sync(self, proc: subprocess.Popen, timeout: float = 3.0) -> bool:
        """
        Wait for process to terminate and confirm reaping.
        If the process does not exit within timeout, attempts force kill and retries wait.
        Returns True if process is confirmed terminated/reaped, False otherwise.
        """
        if proc is None:
            return True

        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                f"[{self.session_id}] Subprocess PID {proc.pid} did not exit within {timeout}s; "
                f"attempting force kill"
            )
            try:
                proc.kill()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Force kill on PID {proc.pid} raised: {e}")

            try:
                proc.wait(timeout=1.0)
                return True
            except Exception as e:
                logger.error(
                    f"[{self.session_id}] Subprocess PID {proc.pid} failed to terminate after force kill: {e}"
                )
                return False
        except Exception as e:
            logger.error(f"[{self.session_id}] Error reaping subprocess PID {proc.pid}: {e}")
            return proc.poll() is not None

    def _kill_process(self, proc: subprocess.Popen) -> threading.Thread:
        """
        Kill and reap a subprocess safely without blocking the event loop.
        Sends termination signal immediately, then delegates confirmed wait() to a background daemon thread.
        """
        try:
            proc.kill()
            logger.info(f"[{self.session_id}] TTS subprocess (pid={proc.pid}) killed successfully")
        except Exception as e:
            logger.warning(f"[{self.session_id}] Failed to kill TTS subprocess: {e}")

        reaper = threading.Thread(
            target=self._reap_process_sync,
            args=(proc, 3.0),
            daemon=True,
            name=f"tts-reaper-{proc.pid}",
        )
        reaper.start()
        return reaper

    async def close(self) -> None:
        """
        Close the session, cancelling any active synthesis and releasing temp files.
        Subsequent synthesis requests will be rejected. Idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self.cancel()

        # Wait for any active synthesis to finish and release locks before cleaning files
        async with self._get_synth_lock():
            pass

        # Clean any remaining temp files tracked by the session in executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._cleanup_all_temp_files)

        # Close persistent HTTP client connection pool if not externally shared
        client = None
        with self._lock:
            if not self._shared_client:
                client = self._http_client
            self._http_client = None

        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Error closing HTTP client: {e}")

        logger.info(f"[{self.session_id}] TTSSession closed cleanly")

    def _cleanup_all_temp_files(self) -> None:
        """Unlink all tracked temp files whose processes have terminated."""
        with self._lock:
            # Check if active process is still running
            if self._current_process is not None and self._current_process.poll() is None:
                logger.warning(
                    f"[{self.session_id}] Subprocess PID {self._current_process.pid} still alive during close; "
                    f"attempting reap before temp file cleanup"
                )
                self._reap_process_sync(self._current_process, timeout=3.0)

            if self._current_process is not None and self._current_process.poll() is None:
                logger.error(
                    f"[{self.session_id}] Cannot cleanup temp files: subprocess PID "
                    f"{self._current_process.pid} could not be terminated."
                )
                return

            files = list(self._active_temp_files)
            self._active_temp_files.clear()

        for path in files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] Failed to unlink temp file {path}: {e}")

    def _spawn_worker(
        self, script_path: str, wav_path: str, req_generation: int
    ) -> Optional[subprocess.Popen]:
        """
        Worker function executed inside thread executor to spawn subprocess.
        Handles race conditions where cancellation occurs before, during, or after Popen.
        NEVER holds self._lock across Popen so event loop is never blocked.
        """
        with self._lock:
            if self._cancelled or self._closed or req_generation != self._cancel_generation:
                return None

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to spawn TTS subprocess: {e}")
            return None

        proc_to_kill = None
        with self._lock:
            if self._cancelled or self._closed or req_generation != self._cancel_generation:
                proc_to_kill = proc
            else:
                self._current_process = proc

        if proc_to_kill is not None:
            logger.info(
                f"[{self.session_id}] Cancellation race detected during spawn (gen {req_generation} "
                f"vs {self._cancel_generation}); terminating PID {proc_to_kill.pid}"
            )
            try:
                proc_to_kill.kill()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Failed to kill PID {proc_to_kill.pid}: {e}")
            self._reap_process_sync(proc_to_kill, timeout=3.0)
            return None

        return proc

    async def stream_tts_audio(
        self,
        text: str,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to WAV using a session-isolated killable subprocess.
        Enforces sequential synthesis per TTSSession and validates request generation
        so queued synthesis cannot speak stale audio after cancellation.
        """
        eff_session_id = session_id or self.session_id

        # Capture request generation before waiting for lock
        with self._lock:
            if self._closed:
                logger.warning(f"[{eff_session_id}] Attempted TTS synthesis on closed TTSSession")
                return
            req_generation = self._cancel_generation

        if not text or not text.strip():
            return

        with self._lock:
            if self._closed:
                logger.warning(f"[{eff_session_id}] TTSSession closed while waiting for synthesis")
                return
            if req_generation != self._cancel_generation:
                logger.info(
                    f"[{eff_session_id}] TTS request generation {req_generation} superseded by "
                    f"cancellation generation {self._cancel_generation}; dropping stale queued synthesis"
                )
                return
            self._cancelled = False

        logger.debug(f"[{eff_session_id}] TTS synthesizing: '{text[:60]}{'...' if len(text) > 60 else ''}'")

        # Check if OpenAI is candidate: either configured as openai with API key, or already committed to openai
        use_openai = (self.provider == "openai" and bool(OPENAI_API_KEY))
        if self._committed_provider == "openai":
            use_openai = True
        elif self._committed_provider == "sapi5":
            use_openai = False

        if use_openai:
            logger.debug(f"[{eff_session_id}] Attempting OpenAI TTS ({self.openai_model}, voice={self.openai_voice})")
            try:
                openai_wav = await self._synthesize_openai(text, req_generation, eff_session_id)
                with self._lock:
                    if self._cancelled or req_generation != self._cancel_generation:
                        return
                if openai_wav:
                    self._committed_provider = "openai"
                    logger.info(f"[{eff_session_id}] ✅ OpenAI TTS synthesized {len(openai_wav)} bytes WAV")
                    yield openai_wav
                    return
            except asyncio.CancelledError:
                self.cancel()
                return
            except Exception as e:
                logger.warning(f"[{eff_session_id}] OpenAI TTS failure: {e}")

            with self._lock:
                if self._cancelled or req_generation != self._cancel_generation:
                    return

            # If this turn already committed to OpenAI, NEVER fall back to SAPI5!
            # Falling back mid-response would mix robotic Hazel with neural Marin.
            if self._committed_provider == "openai":
                logger.error(
                    f"[{eff_session_id}] OpenAI TTS failed for subsequent sentence in turn; "
                    f"suppressing SAPI5 fallback to prevent voice mixing within one response."
                )
                return

        # SAPI5 fallback (only if not committed to OpenAI):
        async with self._get_synth_lock():
            with self._lock:
                if self._closed:
                    logger.warning(f"[{eff_session_id}] TTSSession closed while waiting for synthesis lock")
                    return
                # Verify request generation is still valid after acquiring lock
                if req_generation != self._cancel_generation:
                    logger.info(
                        f"[{eff_session_id}] TTS request generation {req_generation} superseded by "
                        f"cancellation generation {self._cancel_generation}; dropping stale queued synthesis"
                    )
                    return
                self._cancelled = False
                self._committed_provider = "sapi5"

            tmp_script = None
            tmp_wav = None
            proc: Optional[subprocess.Popen] = None
            loop = asyncio.get_running_loop()

            try:
                # Create temp files
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, encoding='utf-8'
                ) as f:
                    tmp_script = f.name
                    f.write(_tts_worker_script(text, tmp_script + ".wav", RATE, VOLUME, voice_name=self.voice_name))

                tmp_wav = tmp_script + ".wav"

                with self._lock:
                    self._active_temp_files.add(tmp_script)
                    self._active_temp_files.add(tmp_wav)

                # Spawn subprocess safely in executor thread (lock not held during Popen)
                try:
                    proc = await loop.run_in_executor(
                        None,
                        self._spawn_worker,
                        tmp_script,
                        tmp_wav,
                        req_generation,
                    )
                except asyncio.CancelledError:
                    self.cancel()
                    raise

                if proc is None:
                    # Was cancelled or closed before/during spawn
                    logger.info(f"[{eff_session_id}] TTS spawn aborted (cancelled, superseded, or closed)")
                    return

                # Wait for subprocess completion in executor thread without blocking event loop
                try:
                    await loop.run_in_executor(None, proc.wait)
                except asyncio.CancelledError:
                    self.cancel()
                    raise

                # Process exit code verification
                if proc.returncode != 0:
                    with self._lock:
                        was_cancelled = self._cancelled or (req_generation != self._cancel_generation)
                    if was_cancelled:
                        logger.info(f"[{eff_session_id}] TTS interrupted — process terminated with code {proc.returncode}")
                    else:
                        logger.warning(f"[{eff_session_id}] TTS subprocess failed with exit code: {proc.returncode}")
                    return

                # WAV output verification
                if not os.path.exists(tmp_wav):
                    logger.warning(f"[{eff_session_id}] TTS output WAV missing: {tmp_wav}")
                    return

                wav_size = os.path.getsize(tmp_wav)
                if wav_size == 0:
                    logger.warning(f"[{eff_session_id}] TTS output WAV is empty (0 bytes): {tmp_wav}")
                    return

                # Read WAV and yield
                with open(tmp_wav, 'rb') as f:
                    wav_bytes = f.read()

                if wav_bytes:
                    yield wav_bytes

            except asyncio.CancelledError:
                self.cancel()
                raise
            except Exception as e:
                logger.error(f"[{eff_session_id}] TTS synthesis error: {e}")
                raise
            finally:
                # Ensure process is confirmed terminated and reaped before unlinking temp files
                reaped = False
                if proc is not None:
                    try:
                        reaped = await asyncio.shield(
                            loop.run_in_executor(None, self._reap_process_sync, proc, 3.0)
                        )
                    except Exception as e:
                        logger.error(f"[{eff_session_id}] Exception while awaiting process reap: {e}")
                        reaped = (proc.poll() is not None)
                else:
                    reaped = True

                with self._lock:
                    if self._current_process is proc:
                        self._current_process = None

                # Do NOT unlink temporary files unless process termination and reaping have been confirmed!
                if reaped:
                    for path in [tmp_script, tmp_wav]:
                        if path:
                            with self._lock:
                                self._active_temp_files.discard(path)
                            if os.path.exists(path):
                                try:
                                    os.unlink(path)
                                except Exception as e:
                                    logger.warning(f"[{eff_session_id}] Failed to unlink temp file {path}: {e}")
                else:
                    logger.error(
                        f"[{eff_session_id}] Temporary files NOT unlinked: subprocess PID "
                        f"{getattr(proc, 'pid', 'unknown')} failed to terminate and be reaped. "
                        f"Retaining files in active tracking."
                    )


# Backwards compatibility module wrappers
def cancel_current_tts():
    """
    Legacy module-level hook for backwards compatibility.
    Individual sessions should call session.cancel().
    """
    logger.debug("Global cancel_current_tts() called (legacy hook)")


async def stream_tts_audio(
    text: str,
    session_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """
    Legacy module-level synthesis wrapper.
    Spawns an ephemeral TTSSession for standalone use.
    """
    session = TTSSession(session_id=session_id)
    try:
        async for chunk in session.stream_tts_audio(text, session_id=session_id):
            yield chunk
    finally:
        await session.close()
