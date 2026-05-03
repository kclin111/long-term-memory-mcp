from __future__ import annotations


IMPORTANT_KEYWORDS = {
    "decide",
    "decided",
    "decision",
    "prefer",
    "prefers",
    "preference",
    "remember",
    "important",
    "correction",
    "correct",
    "project",
    "task",
    "commit",
    "error",
    "workaround",
    "bug",
    "design",
    "schema",
    "implement",
    "實作",
    "決定",
    "偏好",
    "記住",
    "重要",
    "修正",
    "錯誤",
    "專案",
    "設計",
    "架構",
}


def score_importance(content: str) -> float:
    text = content.lower()
    hits = sum(1 for keyword in IMPORTANT_KEYWORDS if keyword.lower() in text)
    base = min(0.35 + hits * 0.15, 0.95)
    if len(content) > 800:
        base = min(base + 0.1, 0.95)
    return round(base, 3)


def score_novelty(content: str) -> float:
    # M2 placeholder. Later releases compare against SessionBuffer, LanceDB,
    # entity states, and recently seen events.
    if len(content.strip()) < 20:
        return 0.2
    return 0.7
