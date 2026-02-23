from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import models
from deps import get_db

router = APIRouter(tags=["prompt-configs"])


class PromptConfigUpsert(BaseModel):
    key: str
    system_prompt: str
    user_prompt: str
    is_active: bool = True


class PromptConfigOut(BaseModel):
    id: int
    key: str
    system_prompt: str
    user_prompt: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


@router.get("/prompt-configs", response_model=list[PromptConfigOut])
def list_prompt_configs(db: Session = Depends(get_db)):
    return db.query(models.PromptConfig).order_by(models.PromptConfig.key.asc()).all()


@router.get("/prompt-configs/{key}", response_model=PromptConfigOut)
def get_prompt_config(key: str, db: Session = Depends(get_db)):
    config = db.query(models.PromptConfig).filter(models.PromptConfig.key == key).first()
    if config is None:
        raise HTTPException(status_code=404, detail="Prompt config not found")
    return config


@router.put("/prompt-configs/{key}", response_model=PromptConfigOut)
def upsert_prompt_config(key: str, payload: PromptConfigUpsert, db: Session = Depends(get_db)):
    if payload.key != key:
        raise HTTPException(status_code=400, detail="Path key and payload key must match")

    config = db.query(models.PromptConfig).filter(models.PromptConfig.key == key).first()
    if config is None:
        config = models.PromptConfig(
            key=payload.key,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
            is_active=payload.is_active,
        )
        db.add(config)
    else:
        config.system_prompt = payload.system_prompt
        config.user_prompt = payload.user_prompt
        config.is_active = payload.is_active

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Prompt key must be unique")

    db.refresh(config)
    return config


@router.delete("/prompt-configs/{key}", status_code=204)
def delete_prompt_config(key: str, db: Session = Depends(get_db)):
    config = db.query(models.PromptConfig).filter(models.PromptConfig.key == key).first()
    if config is None:
        raise HTTPException(status_code=404, detail="Prompt config not found")

    db.delete(config)
    db.commit()
    return None
