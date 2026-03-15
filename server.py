"""
Real-Time Voice AI Backend
WebSocket gateway + STT → LLM → TTS pipeline
"""
import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging

import uuid
from typing import Optional


import websockets
from websockets.server import WebSocketServerProtocol

from pipeline import VoicePipeline
from session import SessionManager

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

session_manager = SessionManager()


async def handle_client(websocket):
    """Handle a new WebSocket client connection."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Client connected from {websocket.remote_address}")

    pipeline = VoicePipeline(session_id=session_id, websocket=websocket)
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

    # RAG warmup — loads sentence-transformers model + embeds knowledge base
    logger.info("Warming up RAG...")
    try:
        from rag import init_rag
        await init_rag()
        logger.info("RAG warmup complete")
    except Exception as e:
        logger.warning(f"RAG warmup failed (non-critical): {e}")


async def main():
    await warmup()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8765"))

    logger.info(f"Starting Voice AI WebSocket server on ws://{host}:{port}")

    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,  # 10MB max message
        ping_interval=20,
        ping_timeout=10,
    ):
        logger.info("Server ready. Waiting for connections...")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
