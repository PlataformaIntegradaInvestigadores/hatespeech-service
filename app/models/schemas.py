from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContentType = Literal["post", "comment"]
Reason = Literal["hidden_by_model", "reported", "deferred", "review_band"]
Visibility = Literal["visible", "hidden", "blocked"]
ReviewStatus = Literal["none", "pending", "resolved"]


class _In(BaseModel):
    # social-service usa ids enteros; se aceptan y se guardan como texto.
    model_config = ConfigDict(coerce_numbers_to_str=True, str_strip_whitespace=True)


class ClassifyRequest(_In):
    text: str = Field(min_length=1, max_length=10000)
    content_type: ContentType
    content_id: str = Field(min_length=1, max_length=64)
    author_id: str | None = Field(default=None, max_length=64)
    explain: bool = True


class TokenSaliency(BaseModel):
    token: str
    saliency: float
    start: int
    end: int


class Explanation(BaseModel):
    method: str | None
    top_tokens: list[TokenSaliency]


class ModelInfo(BaseModel):
    backend: str
    version: str


class Thresholds(BaseModel):
    class_threshold: float
    hide_threshold: float
    hitl_tau: float
    review_band_min: float | None = None
    report_threshold: int


class ClassifyResponse(BaseModel):
    case_id: int
    content_type: ContentType
    content_id: str
    label: Literal["odio", "no_odio"]
    p_hate: float
    confidence: float
    decision: Literal["AUTO", "DERIVAR"]
    action: Literal["visible", "hidden"]
    visibility: Visibility
    review_status: ReviewStatus
    reasons: list[Reason]
    explanation: Explanation
    model: ModelInfo
    thresholds: Thresholds


class ReportRequest(_In):
    content_type: ContentType
    content_id: str = Field(min_length=1, max_length=64)
    reporter_id: str | None = Field(default=None, max_length=64)
    reason: str | None = Field(default=None, max_length=255)
    text: str | None = Field(default=None, max_length=10000)
    author_id: str | None = Field(default=None, max_length=64)


class ReportResponse(BaseModel):
    case_id: int
    content_type: ContentType
    content_id: str
    created: bool
    report_count: int
    report_threshold: int
    threshold_reached: bool
    visibility: Visibility
    review_status: ReviewStatus
    # social-service copia `reasons` al post/comentario solo si la clave viene
    # en el JSON (apply_case_state). Sin ella, un reporte que oculta el
    # contenido actualiza la visibilidad y deja los motivos viejos.
    reasons: list[Reason]


class ModerationStatus(BaseModel):
    content_type: ContentType
    content_id: str
    classified: bool
    visibility: Visibility
    show_warning: bool
    sort_last: bool
    review_status: ReviewStatus
    reasons: list[Reason]
    report_count: int
    label: str | None = None
    p_hate: float | None = None


class DecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    moderator_id: str
    decision: str
    notes: str | None
    previous_visibility: str
    previous_review_status: str
    previous_reasons: list[str]
    new_visibility: str
    p_hate_at_decision: float | None
    created_at: datetime


class ReviewCase(BaseModel):
    id: int
    content_type: ContentType
    content_id: str
    author_id: str | None
    text: str | None
    label: str | None
    p_hate: float | None
    confidence: float | None
    decision: str | None
    explanation: Explanation
    visibility: Visibility
    review_status: ReviewStatus
    reasons: list[Reason]
    report_count: int
    model_version: str | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None


class ReviewCaseDetail(ReviewCase):
    decisions: list[DecisionOut]
    report_reasons: list[str]


class ReviewQueuePage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ReviewCase]


class DecisionRequest(BaseModel):
    decision: Literal["approve", "keep_hidden", "block"]
    notes: str | None = Field(default=None, max_length=2000)


class DecisionResponse(BaseModel):
    case: ReviewCase
    decision: DecisionOut
