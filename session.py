"""
Session management - tracks active voice sessions.
"""

import logging
from typing import Dict

logger = logging.getLogger("session")


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, object] = {}

    def add(self, session_id: str, pipeline):
        self._sessions[session_id] = pipeline
        logger.info(f"Session added: {session_id} | Total active: {len(self._sessions)}")

    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)
        logger.info(f"Session removed: {session_id} | Total active: {len(self._sessions)}")

    def get(self, session_id: str):
        return self._sessions.get(session_id)

    def count(self) -> int:
        return len(self._sessions)