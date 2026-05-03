from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict


@dataclass(frozen=True)
class SessionPage:
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    turn_index: int


class SessionBuffer:
    def __init__(self, max_pages: int = 10) -> None:
        self.max_pages = max_pages
        self._pages: DefaultDict[str, deque[SessionPage]] = defaultdict(lambda: deque(maxlen=max_pages))

    def add(self, page: SessionPage) -> None:
        self._pages[page.session_id].append(page)

    def recent(self, session_id: str) -> list[SessionPage]:
        return list(self._pages.get(session_id, ()))

