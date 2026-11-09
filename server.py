"""
Real-Time Voice AI Backend
WebSocket gateway + STT → LLM → TTS pipeline
"""
import os
import threading
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import uuid
from typing import Optional

import httpx
import websockets

from pipeline import VoicePipeline
from session import SessionManager

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

session_manager = SessionManager()
_warmed_tts_client: Optional[httpx.AsyncClient] = None


async def handle_client(websocket):
    """Handle a new WebSocket client connection."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Client connected from {websocket.remote_address}")

    pipeline = VoicePipeline(session_id=session_id, websocket=websocket, tts_http_client=_warmed_tts_client)
    session_manager.add(session_id, pipeline)

    try:
        await pipeline.run(websocket)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"[{session_id}] Client disconnected cleanly")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"[{session_id}] Connection closed with error: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
    finally:
        await pipeline.cleanup()
        session_manager.remove(session_id)
        logger.info(f"[{session_id}] Session cleaned up")


async def warmup():
    """Warm up STT and RAG on server start to avoid first-query delays."""
    import io, wave
    import numpy as np

    # STT warmup
    logger.info("Warming up STT model...")
    silence = np.zeros(8000, dtype=np.int16).tobytes()
    try:
        from stt import transcribe_audio
        await transcribe_audio(silence, session_id="warmup")
        logger.info("STT warmup complete")
    except Exception as e:
        logger.warning(f"STT warmup failed (non-critical): {e}")

    # RAG warmup : loads sentence-transformers model + embeds knowledge base
    logger.info("Warming up RAG...")
    try:
        from rag import init_rag
        await init_rag()
        logger.info("RAG warmup complete")
    except Exception as e:
        logger.warning(f"RAG warmup failed (non-critical): {e}")

    # TTS warmup : warms up persistent OpenAI TTS client and pre-resolves DNS/TLS
    global _warmed_tts_client
    logger.info("Warming up TTS...")
    try:
        from config import TTS_PROVIDER, OPENAI_API_KEY
        if TTS_PROVIDER == "openai" and bool(OPENAI_API_KEY):
            logger.info("Warming up OpenAI TTS connection pool...")
            timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
            _warmed_tts_client = httpx.AsyncClient(timeout=timeout)
            from tts import TTSSession
            tts_warmup = TTSSession(session_id="warmup", http_client=_warmed_tts_client)
            async for _ in tts_warmup.stream_tts_audio("Good day."):
                pass
            logger.info("OpenAI TTS warmup complete (TLS connection active and pooled)")
    except Exception as e:
        logger.warning(f"TTS warmup failed (non-critical): {e}")


def start_http_server(host: str = "127.0.0.1", port: int = 8000):
    """Serve frontend index.html using Python standard library http.server in a daemon thread.
    Restricted exclusively to local loopback (127.0.0.1:8000) with a strict frontend allowlist.
    Only frontend files (index.html, /) are served.
    Directory listing is disabled.
    All project files (.env, Python sources, .git, tests, configs, etc.) return 404/403.
    """
    web_dir = Path(__file__).parent.resolve()

    class SecureFrontendHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def do_GET(self):
            # Normalize request path ignoring query string and fragment
            clean_path = urlsplit(self.path).path.strip()
            # Explicit allowlist: only serve root / or /index.html
            if clean_path in ("/", "/index.html"):
                self.path = "/index.html"
                return super().do_GET()
            elif clean_path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            # Explicitly reject all other paths (.env, .py, .git, etc.) with 404
            self.send_error(404, "Not Found")

        def do_HEAD(self):
            clean_path = urlsplit(self.path).path.strip()
            if clean_path in ("/", "/index.html"):
                self.path = "/index.html"
                return super().do_HEAD()
            elif clean_path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404, "Not Found")

        def list_directory(self, path):
            # Explicitly disable directory listing
            self.send_error(403, "Directory listing forbidden")
            return None

        def log_message(self, format, *args):
            logger.debug(f"HTTP: {format % args}")

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), SecureFrontendHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="frontend-http")
        thread.start()
        logger.info(f"Frontend HTTP server running exclusively at http://127.0.0.1:{port}")
        return httpd
    except OSError as e:
        logger.warning(
            f"Could not bind frontend HTTP server to 127.0.0.1:{port} ({e}). "
            f"If already running, you can open http://127.0.0.1:{port} directly."
        )
        return None


async def main():
    await warmup()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    http_port = int(os.getenv("HTTP_PORT", "8000"))

    start_http_server(host="127.0.0.1", port=http_port)
    logger.info(f"Starting Voice AI WebSocket server on ws://{host}:{port}")

    try:
        async with websockets.serve(
            handle_client,
            host,
            port,
            origins=["http://127.0.0.1:8000", "http://localhost:8000"],
            max_size=1 * 1024 * 1024,  # 1MB max message
            ping_interval=20,
            ping_timeout=10,
        ):
            logger.info("Server ready. Waiting for connections...")
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                logger.info("Server shutdown requested")
    finally:
        if _warmed_tts_client is not None and not _warmed_tts_client.is_closed:
            await _warmed_tts_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
