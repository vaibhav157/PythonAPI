import json
import math
import os
import re
import threading
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from fastapi import HTTPException
from openai import OpenAI
from sqlalchemy.orm import Session

from services.prompt_config_service import get_prompt_bundle, render_user_prompt

NICKNAME_TO_CANONICAL = {
    "andy": "andrew",
    "drew": "andrew",
    "mike": "michael",
    "mick": "michael",
    "tony": "anthony",
    "alex": "alexander",
    "sam": "samuel",
    "jeff": "jeffrey",
    "bill": "william",
    "will": "william",
    "bob": "robert",
    "rob": "robert",
    "bobby": "robert",
    "kate": "katherine",
    "katie": "katherine",
    "liz": "elizabeth",
    "beth": "elizabeth",
    "jon": "jonathan",
    "johnny": "john",
}

ALLOWED_MATCH_RESULTS = {"match", "partial_match", "no_match"}
PERSON_ACCOUNT_PROMPT_KEY = "person_account_match"
_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()


def _resolve_openai_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _resolve_embedding_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _extract_response_text(raw_response: dict[str, Any]) -> str:
    output = raw_response.get("output", [])
    texts: list[str] = []

    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)

    return "\n".join(texts).strip()


def _parse_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _soundex(value: str) -> str:
    if not value:
        return ""

    text = re.sub(r"[^A-Za-z]", "", value.upper())
    if not text:
        return ""

    mappings = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        "L": "4",
        **dict.fromkeys(list("MN"), "5"),
        "R": "6",
    }

    first = text[0]
    encoded = [first]
    prev = mappings.get(first, "")

    for char in text[1:]:
        code = mappings.get(char, "")
        if code != prev:
            if code:
                encoded.append(code)
            prev = code

    result = "".join(encoded)
    return (result + "000")[:4]


def _fuzzy_score(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio() * 100, 2)


def _is_fuzzy_match(a: str, b: str, threshold: float = 82.0) -> bool:
    return _fuzzy_score(a, b) >= threshold


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    score = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, score))


def _semantic_percent(similarity: float) -> float:
    return round(((similarity + 1.0) / 2.0) * 100.0, 2)


def _embedding_with_cache(client: OpenAI, model: str, text: str) -> list[float]:
    key = (model, text)

    with _EMBEDDING_CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached

    embedding_response = client.embeddings.create(model=model, input=text)
    vector = embedding_response.data[0].embedding

    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE[key] = vector

    return vector


def normalize_text(value: str | None) -> str:
    return (value or "").strip().casefold()


def normalize_postcode(value: str | None) -> str:
    return re.sub(r"\s+", "", normalize_text(value)).upper()


def normalize_phone(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    # Compare on local significant suffix when country code format varies.
    return digits[-10:] if len(digits) > 10 else digits


def normalize_dob(value: str | None) -> str:
    if not value:
        return ""

    candidate = value.strip()
    patterns = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
    for pattern in patterns:
        try:
            return datetime.strptime(candidate, pattern).date().isoformat()
        except ValueError:
            continue

    return candidate


def canonical_first_name(name: str | None) -> str:
    normalized = normalize_text(name)
    return NICKNAME_TO_CANONICAL.get(normalized, normalized)


def build_normalized_record(record: dict[str, str]) -> dict[str, str]:
    return {
        "first_name": normalize_text(record.get("first_name")),
        "last_name": normalize_text(record.get("last_name")),
        "dob": normalize_dob(record.get("dob")),
        "address": normalize_text(record.get("address")),
        "postcode": normalize_postcode(record.get("postcode")),
        "email": normalize_text(record.get("email")),
        "phone": normalize_phone(record.get("phone")),
        "first_name_canonical": canonical_first_name(record.get("first_name")),
    }


def _name_text(record: dict[str, str]) -> str:
    return " ".join(
        part for part in [record.get("first_name_canonical", ""), record.get("last_name", "")] if part
    ).strip()


def _identity_text(record: dict[str, str]) -> str:
    parts = [
        record.get("first_name_canonical", ""),
        record.get("last_name", ""),
        record.get("dob", ""),
        record.get("address", ""),
        record.get("postcode", ""),
        record.get("email", ""),
        record.get("phone", ""),
    ]
    return " | ".join(parts)


def build_match_signals(person_record: dict[str, str], account_record: dict[str, str]) -> dict[str, Any]:
    person_first = person_record.get("first_name", "")
    account_first = account_record.get("first_name", "")
    person_first_canonical = person_record.get("first_name_canonical", "")
    account_first_canonical = account_record.get("first_name_canonical", "")

    person_last = person_record.get("last_name", "")
    account_last = account_record.get("last_name", "")

    person_address = person_record.get("address", "")
    account_address = account_record.get("address", "")

    first_name_fuzzy_score = _fuzzy_score(person_first_canonical, account_first_canonical)
    last_name_fuzzy_score = _fuzzy_score(person_last, account_last)
    address_fuzzy_score = _fuzzy_score(person_address, account_address)

    return {
        "first_name_exact": person_first == account_first,
        "first_name_canonical_exact": person_first_canonical == account_first_canonical,
        "first_name_fuzzy_score": first_name_fuzzy_score,
        "first_name_fuzzy_match": _is_fuzzy_match(person_first_canonical, account_first_canonical),
        "first_name_soundex_person": _soundex(person_first_canonical),
        "first_name_soundex_account": _soundex(account_first_canonical),
        "first_name_soundex_match": _soundex(person_first_canonical) == _soundex(account_first_canonical),
        "last_name_fuzzy_score": last_name_fuzzy_score,
        "last_name_fuzzy_match": _is_fuzzy_match(person_last, account_last),
        "last_name_soundex_person": _soundex(person_last),
        "last_name_soundex_account": _soundex(account_last),
        "last_name_soundex_match": _soundex(person_last) == _soundex(account_last),
        "address_fuzzy_score": address_fuzzy_score,
        "address_fuzzy_match": _is_fuzzy_match(person_address, account_address, threshold=78.0),
        "address_soundex_person": _soundex(person_address),
        "address_soundex_account": _soundex(account_address),
        "address_soundex_match": _soundex(person_address) == _soundex(account_address),
    }


def build_semantic_signals(
    client: OpenAI,
    person_record: dict[str, str],
    account_record: dict[str, str],
    embedding_model: str,
) -> dict[str, Any]:
    person_name_vector = _embedding_with_cache(client, embedding_model, _name_text(person_record))
    account_name_vector = _embedding_with_cache(client, embedding_model, _name_text(account_record))

    person_identity_vector = _embedding_with_cache(client, embedding_model, _identity_text(person_record))
    account_identity_vector = _embedding_with_cache(client, embedding_model, _identity_text(account_record))

    name_similarity = _cosine_similarity(person_name_vector, account_name_vector)
    identity_similarity = _cosine_similarity(person_identity_vector, account_identity_vector)

    return {
        "semantic_embedding_model": embedding_model,
        "name_semantic_similarity": round(name_similarity, 4),
        "name_semantic_score": _semantic_percent(name_similarity),
        "identity_semantic_similarity": round(identity_similarity, 4),
        "identity_semantic_score": _semantic_percent(identity_similarity),
    }


def score_person_vs_account(
    db: Session,
    person_original: dict[str, str],
    account_original: dict[str, str],
    model: str | None,
    include_semantic_signals: bool = False,
    embedding_model: str | None = None,
) -> tuple[str, float, str, dict[str, Any], dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    person_normalized = build_normalized_record(person_original)
    account_normalized = build_normalized_record(account_original)
    match_signals = build_match_signals(person_normalized, account_normalized)

    client = OpenAI(api_key=api_key)

    if include_semantic_signals:
        resolved_embedding_model = _resolve_embedding_model(embedding_model)
        semantic_signals = build_semantic_signals(
            client=client,
            person_record=person_normalized,
            account_record=account_normalized,
            embedding_model=resolved_embedding_model,
        )
        match_signals.update(semantic_signals)

    resolved_model = _resolve_openai_model(model)
    system_prompt, user_prompt_template = get_prompt_bundle(db, PERSON_ACCOUNT_PROMPT_KEY)
    user_prompt = render_user_prompt(
        user_prompt_template,
        person_original=json.dumps(person_original, ensure_ascii=True),
        person_normalized=json.dumps(person_normalized, ensure_ascii=True),
        account_original=json.dumps(account_original, ensure_ascii=True),
        account_normalized=json.dumps(account_normalized, ensure_ascii=True),
    )

    # Always append deterministic signals; semantic signals are appended when requested.
    user_prompt = (
        f"{user_prompt}\n"
        f"Derived match signals: {json.dumps(match_signals, ensure_ascii=True)}"
    )

    try:
        response = client.responses.create(
            model=resolved_model,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )

        raw_response = response.model_dump()
        content_text = _extract_response_text(raw_response)
        payload = _parse_json_payload(content_text)

        match_result = str(payload.get("match_result", "no_match")).strip().lower()
        if match_result not in ALLOWED_MATCH_RESULTS:
            match_result = "no_match"

        confidence = float(payload.get("confidence_percentage", 0.0))
        confidence = max(0.0, min(100.0, confidence))

        description = str(
            payload.get("description")
            or payload.get("detailed_description")
            or payload.get("reason")
            or "No description provided"
        )

        return match_result, confidence, description, raw_response, match_signals
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI comparison failed: {exc}") from exc
