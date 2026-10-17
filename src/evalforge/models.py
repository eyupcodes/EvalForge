"""Core data models shared by judges, runners and reports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Case(BaseModel):
    """One evaluation item: what the model was asked and what it answered."""

    id: str
    input: str
    output: str
    reference: str | None = None
    context: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """A judge's decision about one case."""

    case_id: str
    judge: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
