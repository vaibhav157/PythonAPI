from typing import Any

from fastapi import APIRouter, Depends
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from deps import get_db
from services.person_account_match_service import score_person_vs_account

router = APIRouter(tags=["person-account-match"])


class PersonRecord(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices("first_name", "FirstName"))
    last_name: str = Field(validation_alias=AliasChoices("last_name", "LastName"))
    dob: str = Field(validation_alias=AliasChoices("dob", "DoB", "DOB"))
    address: str = Field(validation_alias=AliasChoices("address", "Address"))
    postcode: str = Field(validation_alias=AliasChoices("postcode", "Postcode", "PostalCode"))
    email: str = Field(validation_alias=AliasChoices("email", "Email"))
    phone: str = Field(validation_alias=AliasChoices("phone", "Phone"))

    model_config = ConfigDict(populate_by_name=True)


class AccountRecord(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices("first_name", "FirstName"))
    last_name: str = Field(validation_alias=AliasChoices("last_name", "LastName"))
    dob: str = Field(validation_alias=AliasChoices("dob", "DoB", "DOB"))
    address: str = Field(validation_alias=AliasChoices("address", "Address"))
    postcode: str = Field(validation_alias=AliasChoices("postcode", "Postcode", "PostalCode"))
    email: str = Field(validation_alias=AliasChoices("email", "Email"))
    phone: str = Field(validation_alias=AliasChoices("phone", "Phone"))

    model_config = ConfigDict(populate_by_name=True)


class PersonAccountMatchRequest(BaseModel):
    person: PersonRecord
    accounts: list[AccountRecord] = Field(min_length=1)
    openai_model: str | None = None
    include_raw_output: bool = False
    include_semantic_signals: bool = False
    embedding_model: str | None = None


class AccountMatchResult(BaseModel):
    account_index: int
    match_result: str
    confidence_percentage: float
    description: str
    match_signals: dict[str, Any]
    raw_output: dict[str, Any] | None = None


class PersonAccountMatchResponse(BaseModel):
    results: list[AccountMatchResult]
    best_match: AccountMatchResult | None


@router.post("/person-account/match", response_model=PersonAccountMatchResponse)
def match_person_to_accounts(payload: PersonAccountMatchRequest, db: Session = Depends(get_db)):
    person_dict = payload.person.model_dump()

    results: list[AccountMatchResult] = []
    for index, account in enumerate(payload.accounts):
        match_result, confidence, description, raw_output, match_signals = score_person_vs_account(
            db=db,
            person_original=person_dict,
            account_original=account.model_dump(),
            model=payload.openai_model,
            include_semantic_signals=payload.include_semantic_signals,
            embedding_model=payload.embedding_model,
        )
        results.append(
            AccountMatchResult(
                account_index=index,
                match_result=match_result,
                confidence_percentage=confidence,
                description=description,
                match_signals=match_signals,
                raw_output=raw_output if payload.include_raw_output else None,
            )
        )

    best_match = max(results, key=lambda item: item.confidence_percentage, default=None)
    return PersonAccountMatchResponse(results=results, best_match=best_match)
