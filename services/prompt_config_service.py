from sqlalchemy.orm import Session

import models

DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "reconcile_match": {
        "system": (
            "You compare two records and return strict JSON with keys: "
            "match_percentage (0 to 100 float) and reasoning (short string)."
        ),
        "user": (
            "Compare these records and score similarity as percentage.\n"
            "Local record: {local_record}\n"
            "Remote record: {remote_record}"
        ),
    },
    "person_account_match": {
        "system": (
            "You are a record-linkage engine. Return strict JSON only with keys: "
            "match_result, confidence_percentage, description. "
            "match_result must be one of: match, partial_match, no_match. "
            "Use nickname equivalence on first names and treat compared values as case-insensitive. "
            "Phone and postcode are already normalized and should be compared as exact strings after normalization."
        ),
        "user": (
            "Compare person and account for identity match.\n"
            "Person original: {person_original}\n"
            "Person normalized: {person_normalized}\n"
            "Account original: {account_original}\n"
            "Account normalized: {account_normalized}"
        ),
    },
}


def get_prompt_bundle(db: Session, key: str) -> tuple[str, str]:
    config = (
        db.query(models.PromptConfig)
        .filter(models.PromptConfig.key == key, models.PromptConfig.is_active.is_(True))
        .first()
    )

    if config:
        return config.system_prompt, config.user_prompt

    default = DEFAULT_PROMPTS.get(key)
    if not default:
        return "", ""
    return default["system"], default["user"]


def render_user_prompt(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except KeyError:
        # Keep template text if a placeholder is missing in stored config.
        return template
