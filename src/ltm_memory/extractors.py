from __future__ import annotations

from datetime import date
import re
import unicodedata

from .scoring import score_importance, score_novelty


TECH_TERMS = {
    "MCP": "concept",
    "SQLite": "tool",
    "FTS5": "tool",
    "LanceDB": "tool",
    "Python": "tool",
    "MemoryOS": "concept",
    "GSW": "concept",
    "Operator": "concept",
    "Reconciler": "concept",
    "EntityNarrative": "concept",
    "PersistentMemory": "concept",
    "OpenQuestion": "concept",
}

PLACE_HINTS = {
    "Museum",
    "Hall",
    "Room",
    "Center",
    "Centre",
    "Building",
    "Campus",
    "Taipei",
    "Hsinchu",
    "Metropolitan",
}

STOP_ENTITIES = {
    "User",
    "The",
    "This",
    "That",
    "And",
    "But",
    "If",
    "MVP",
    "M0",
    "M1",
    "M2",
    "M3",
    "Remember",
    "Important",
}


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().lower()


def classify_event_type(content: str) -> str:
    text = content.lower()
    if any(token in text for token in ("decided", "decision", "決定")):
        return "decision"
    if any(token in text for token in ("prefer", "preference", "偏好", "喜歡")):
        return "preference"
    if any(token in text for token in ("correct", "correction", "修正")):
        return "correction"
    if any(token in text for token in ("error", "bug", "workaround", "錯誤")):
        return "error_workaround"
    if any(token in text for token in ("presented", "explained", "attended", "happened", "發表", "說明")):
        return "episodic_event"
    if any(token in text for token in ("project", "roadmap", "專案", "架構")):
        return "project_state"
    return "interaction"


def summarize_observation(content: str, max_chars: int = 240) -> str:
    normalized = " ".join(content.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _add_entity(
    candidates: dict[tuple[str, str], dict],
    *,
    canonical_name: str,
    entity_type: str,
    role_in_event: str,
    confidence: float,
) -> None:
    key = (_normalize_name(canonical_name), entity_type)
    candidates.setdefault(
        key,
        {
            "canonical_name": canonical_name.strip(),
            "entity_type": entity_type,
            "role_in_event": role_in_event,
            "confidence": confidence,
        },
    )


def extract_entity_candidates(content: str, *, source_type: str) -> list[dict]:
    candidates: dict[tuple[str, str], dict] = {}

    if source_type == "chat":
        _add_entity(
            candidates,
            canonical_name="user",
            entity_type="person",
            role_in_event="speaker",
            confidence=0.9,
        )

    lower_content = content.lower()
    for term, entity_type in TECH_TERMS.items():
        if term.lower() in lower_content:
            _add_entity(
                candidates,
                canonical_name=term,
                entity_type=entity_type,
                role_in_event="mentioned",
                confidence=0.82,
            )

    multiword_parts: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_&.'+-]+(?:\s+[A-Z][A-Za-z0-9_&.'+-]+){1,3}\b", content):
        name = match.group(0).strip()
        if any(part in STOP_ENTITIES for part in name.split()):
            continue
        multiword_parts.update(name.split())
        entity_type = "place" if any(hint in name for hint in PLACE_HINTS) else "person"
        _add_entity(
            candidates,
            canonical_name=name,
            entity_type=entity_type,
            role_in_event="mentioned",
            confidence=0.68,
        )

    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_+-]{2,}\b", content):
        name = match.group(0)
        if name in STOP_ENTITIES or name in multiword_parts:
            continue
        entity_type = "place" if name in PLACE_HINTS else "concept"
        _add_entity(
            candidates,
            canonical_name=name,
            entity_type=entity_type,
            role_in_event="mentioned",
            confidence=0.55,
        )

    return list(candidates.values())


def extract_anchors(content: str) -> list[dict]:
    anchors: dict[tuple[str, str, str], dict] = {}
    full_date_spans: list[tuple[int, int]] = []

    for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", content):
        full_date_spans.append(match.span())
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            normalized = date(year, month, day).isoformat()
        except ValueError:
            continue
        anchors[("time", normalized, "day")] = {
            "anchor_type": "time",
            "label": match.group(0),
            "normalized_value": normalized,
            "granularity": "day",
            "confidence": 0.92,
            "status": "confirmed",
        }

    for match in re.finditer(r"\b(20\d{2})\b", content):
        if any(start <= match.start() < end for start, end in full_date_spans):
            continue
        normalized = match.group(1)
        anchors.setdefault(
            ("time", normalized, "year"),
            {
                "anchor_type": "time",
                "label": normalized,
                "normalized_value": normalized,
                "granularity": "year",
                "confidence": 0.7,
                "status": "candidate",
            },
        )

    for match in re.finditer(
        r"\b(?:at|in)\s+([A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){0,4})",
        content,
    ):
        label = match.group(1).strip().rstrip(".,;:")
        if label in STOP_ENTITIES or re.match(r"20\d{2}", label):
            continue
        granularity = "venue" if any(hint in label for hint in PLACE_HINTS) else "city"
        anchors[("space", _normalize_name(label), granularity)] = {
            "anchor_type": "space",
            "label": label,
            "normalized_value": _normalize_name(label),
            "granularity": granularity,
            "confidence": 0.78,
            "status": "confirmed",
        }

    return list(anchors.values())


def _find_entity(entities: list[dict], name: str) -> dict | None:
    normalized = _normalize_name(name)
    for entity in entities:
        if _normalize_name(entity["canonical_name"]) == normalized:
            return entity
    return None


def extract_roles(content: str, event_type: str, entities: list[dict]) -> list[dict]:
    roles: list[dict] = []
    if _find_entity(entities, "user"):
        roles.append({"entity": "user", "role_type": "speaker", "confidence": 0.86})
        if event_type == "decision":
            roles.append({"entity": "user", "role_type": "decision_maker", "confidence": 0.8})
        if event_type == "preference":
            roles.append({"entity": "user", "role_type": "preference_holder", "confidence": 0.82})

    for entity in entities:
        name = entity["canonical_name"]
        if entity["entity_type"] == "person" and name != "user":
            escaped = re.escape(name)
            if re.search(rf"\b{escaped}\b.*\b(presented|explained|發表|說明)\b", content, flags=re.IGNORECASE):
                roles.append({"entity": name, "role_type": "presenter", "confidence": 0.75})
            elif re.search(rf"\b{escaped}\b.*\b(attended|joined|參加)\b", content, flags=re.IGNORECASE):
                roles.append({"entity": name, "role_type": "attendee", "confidence": 0.72})
    return roles


def extract_states(content: str, event_type: str, entities: list[dict]) -> list[dict]:
    states: list[dict] = []
    if event_type == "decision" and _find_entity(entities, "user"):
        states.append(
            {
                "entity": "user",
                "state_type": "decision",
                "value": summarize_observation(content, max_chars=180),
                "role_type": "decision_maker",
                "confidence": 0.72,
            }
        )
    if event_type == "preference" and _find_entity(entities, "user"):
        states.append(
            {
                "entity": "user",
                "state_type": "preference",
                "value": summarize_observation(content, max_chars=180),
                "role_type": "preference_holder",
                "confidence": 0.74,
            }
        )
    for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+(?:is|was)\s+(nervous|ready|blocked)\b", content):
        entity_name = match.group(1)
        if _find_entity(entities, entity_name):
            states.append(
                {
                    "entity": entity_name,
                    "state_type": "status",
                    "value": match.group(2).lower(),
                    "role_type": None,
                    "confidence": 0.65,
                }
            )
    return states


def extract_actions(content: str, event_type: str, entities: list[dict]) -> list[dict]:
    actor = "user" if _find_entity(entities, "user") else None
    action_type = {
        "decision": "decide",
        "preference": "prefer",
        "correction": "correct",
        "error_workaround": "workaround",
        "project_state": "update",
        "episodic_event": "occur",
    }.get(event_type, "mention")

    for entity in entities:
        name = entity["canonical_name"]
        if entity["entity_type"] == "person" and name != "user":
            escaped = re.escape(name)
            if re.search(rf"\b{escaped}\b.*\bexplained\b", content, flags=re.IGNORECASE):
                actor = name
                action_type = "explain"
                break
            if re.search(rf"\b{escaped}\b.*\bpresented\b", content, flags=re.IGNORECASE):
                actor = name
                action_type = "present"
                break

    object_name = None
    for entity in entities:
        if entity["canonical_name"] not in {actor, "user"}:
            object_name = entity["canonical_name"]
            break

    return [
        {
            "actor": actor,
            "action_type": action_type,
            "action_text": summarize_observation(content, max_chars=180),
            "object": object_name,
            "valence": "positive" if event_type in {"decision", "preference"} else "neutral",
            "confidence": 0.62,
        }
    ]


def propose_open_questions(entities: list[dict], anchors: list[dict], event_type: str) -> list[dict]:
    if event_type == "interaction":
        return []

    has_time = any(anchor["anchor_type"] == "time" for anchor in anchors)
    has_space = any(anchor["anchor_type"] == "space" for anchor in anchors)
    subjects = [entity for entity in entities if entity["canonical_name"] != "user"] or entities[:1]
    questions: list[dict] = []
    related = [entity["canonical_name"] for entity in entities]

    for subject in subjects[:2]:
        name = subject["canonical_name"]
        if not has_time:
            questions.append(
                {
                    "subject": name,
                    "question_type": "when",
                    "question_text": f"When did {name} participate in this event?",
                    "related_entities": related,
                    "priority": 2,
                    "confidence": 0.52,
                }
            )
        if not has_space:
            questions.append(
                {
                    "subject": name,
                    "question_type": "where",
                    "question_text": f"Where did {name} participate in this event?",
                    "related_entities": related,
                    "priority": 2,
                    "confidence": 0.52,
                }
            )
    return questions


def extract_event_proposal(observation: dict) -> dict:
    content = observation["content"]
    event_type = classify_event_type(content)
    entities = extract_entity_candidates(content, source_type=observation["source_type"])
    anchors = extract_anchors(content)
    return {
        "event_type": event_type,
        "summary": summarize_observation(content),
        "structured": {
            "extractor": "deterministic_m2",
            "source_observation_id": observation["id"],
            "staging_note": "actions, roles, states, anchors, and open_questions are reified by IntegrateWorker",
        },
        "importance": score_importance(content),
        "novelty": score_novelty(content),
        "confidence": 0.58,
        "entities": entities,
        "roles": extract_roles(content, event_type, entities),
        "states": extract_states(content, event_type, entities),
        "actions": extract_actions(content, event_type, entities),
        "anchors": anchors,
        "open_questions": propose_open_questions(entities, anchors, event_type),
        "evidence_excerpt": summarize_observation(content, max_chars=500),
    }
