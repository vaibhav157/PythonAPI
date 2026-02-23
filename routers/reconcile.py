import json
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import models
from db import get_remote_engine
from deps import get_db
from services.prompt_config_service import get_prompt_bundle, render_user_prompt

router = APIRouter(tags=["reconcile"])


RECONCILE_PROMPT_KEY = "reconcile_match"


def _resolve_openai_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class ReconcileRequest(BaseModel):
    source_db_url: str
    source_query: str
    source_params: dict[str, Any] = Field(default_factory=dict)
    target_db_url: str
    target_query: str
    match_keys: list[str] = Field(min_length=1)
    target_static_params: dict[str, Any] = Field(default_factory=dict)
    openai_model: str | None = None
    max_source_records: int = Field(default=100, ge=1, le=1000)
    max_target_matches_per_record: int = Field(default=20, ge=1, le=200)


class LocalRecordOut(BaseModel):
    id: int
    source_record_id: str | None
    payload: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class MatchEvaluationOut(BaseModel):
    id: int
    local_record_id: int
    remote_record_id: str | None
    match_percentage: float
    reasoning: str
    remote_payload: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class ReconcileResponse(BaseModel):
    ingested_records: int
    evaluations_created: int
    results: list[MatchEvaluationOut]


def _execute_query(database_url: str, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    remote_engine = get_remote_engine(database_url)
    with remote_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result.fetchall()]


def _build_target_params(
    source_row: dict[str, Any],
    match_keys: list[str],
    static_params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(static_params)
    for key in match_keys:
        if key not in source_row:
            raise HTTPException(
                status_code=400,
                detail=f"Match key '{key}' not found in source query result",
            )
        params[key] = source_row[key]
    return params


def _score_with_openai(
    db: Session,
    local_row: dict[str, Any],
    remote_row: dict[str, Any],
    model: str | None,
) -> tuple[float, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    resolved_model = _resolve_openai_model(model)
    system_prompt, user_prompt_template = get_prompt_bundle(db, RECONCILE_PROMPT_KEY)
    user_prompt = render_user_prompt(
        user_prompt_template,
        local_record=json.dumps(local_row, ensure_ascii=True),
        remote_record=json.dumps(remote_row, ensure_ascii=True),
    )

    client = OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=resolved_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        percentage = float(parsed.get("match_percentage", 0.0))
        percentage = max(0.0, min(100.0, percentage))
        reasoning = str(parsed.get("reasoning", "No reasoning provided"))
        return percentage, reasoning
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI scoring failed: {exc}") from exc


@router.get("/records", response_model=list[LocalRecordOut])
def list_local_records(db: Session = Depends(get_db)):
    return db.query(models.LocalRecord).order_by(models.LocalRecord.id.asc()).all()


@router.get("/matches", response_model=list[MatchEvaluationOut])
def list_match_evaluations(db: Session = Depends(get_db)):
    return db.query(models.MatchEvaluation).order_by(models.MatchEvaluation.id.asc()).all()


@router.post("/reconcile", response_model=ReconcileResponse)
def reconcile(payload: ReconcileRequest, db: Session = Depends(get_db)):
    try:
        source_rows = _execute_query(payload.source_db_url, payload.source_query, payload.source_params)
        source_rows = source_rows[: payload.max_source_records]

        if not source_rows:
            return ReconcileResponse(ingested_records=0, evaluations_created=0, results=[])

        local_records: list[models.LocalRecord] = []
        for row in source_rows:
            source_id = str(row.get("id")) if row.get("id") is not None else None
            local_record = models.LocalRecord(source_record_id=source_id, payload=row)
            db.add(local_record)
            local_records.append(local_record)

        db.flush()

        created_results: list[MatchEvaluationOut] = []

        for index, local_record in enumerate(local_records):
            source_row = source_rows[index]
            target_params = _build_target_params(
                source_row=source_row,
                match_keys=payload.match_keys,
                static_params=payload.target_static_params,
            )
            target_rows = _execute_query(payload.target_db_url, payload.target_query, target_params)
            target_rows = target_rows[: payload.max_target_matches_per_record]

            for target_row in target_rows:
                percentage, reasoning = _score_with_openai(
                    db=db,
                    local_row=source_row,
                    remote_row=target_row,
                    model=payload.openai_model,
                )

                remote_id = str(target_row.get("id")) if target_row.get("id") is not None else None
                match = models.MatchEvaluation(
                    local_record_id=local_record.id,
                    remote_record_id=remote_id,
                    remote_payload=target_row,
                    match_percentage=percentage,
                    reasoning=reasoning,
                )
                db.add(match)
                db.flush()

                created_results.append(
                    MatchEvaluationOut(
                        id=match.id,
                        local_record_id=match.local_record_id,
                        remote_record_id=match.remote_record_id,
                        match_percentage=match.match_percentage,
                        reasoning=match.reasoning,
                        remote_payload=match.remote_payload,
                    )
                )

        db.commit()

        return ReconcileResponse(
            ingested_records=len(local_records),
            evaluations_created=len(created_results),
            results=created_results,
        )
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
