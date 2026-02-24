import json
import os
import re
from datetime import datetime
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


def _resolve_openai_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


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
        "postcode": normalize_postcode(record.get("postcode")),
        "email": normalize_text(record.get("email")),
        "phone": normalize_phone(record.get("phone")),
        "first_name_canonical": canonical_first_name(record.get("first_name")),
    }


def score_person_vs_account(
    db: Session,
    person_original: dict[str, str],
    account_original: dict[str, str],
    model: str | None,
) -> tuple[str, float, str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    person_normalized = build_normalized_record(person_original)
    account_normalized = build_normalized_record(account_original)

    resolved_model = _resolve_openai_model(model)
    system_prompt, user_prompt_template = get_prompt_bundle(db, PERSON_ACCOUNT_PROMPT_KEY)
    user_prompt = render_user_prompt(
        user_prompt_template,
        person_original=json.dumps(person_original, ensure_ascii=True),
        person_normalized=json.dumps(person_normalized, ensure_ascii=True),
        account_original=json.dumps(account_original, ensure_ascii=True),
        account_normalized=json.dumps(account_normalized, ensure_ascii=True),
    )

    client = OpenAI(api_key=api_key)

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

        return match_result, confidence, description, raw_response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI comparison failed: {exc}") from exc
