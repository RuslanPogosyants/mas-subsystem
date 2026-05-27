"""Pydantic schemas for domain entities.

Task, Document, TextChunk, Summary, Term, Quiz, Citation, ResultArtifact.
Matches the data model in VKR thesis section 2.2 and ER diagram (figure 2.9).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Operation(StrEnum):
    """Functional requirements F1-F6 from VKR thesis section 1.3."""

    F1_TRANSCRIBE = "F1"
    F2_OCR = "F2"
    F3_SUMMARIZE = "F3"
    F4_TEST = "F4"
    F5_TERMS = "F5"
    F6_RECOMMEND = "F6"


class DocumentType(StrEnum):
    """Type of the input document medium."""

    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"


class TaskStatus(StrEnum):
    """Task status values (VKR thesis section 2.6, sequence diagram)."""

    PLANNING = "planning"
    RUNNING = "running"
    PARTIAL_READY = "partial_ready"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    document_type: DocumentType
    file_path: str
    original_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    document_id: str
    source_type: Literal["audio", "image", "text", "pdf_extracted"]
    content: str
    chunk_index: int
    confidence: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class SummarySection(BaseModel):
    type: Literal["introduction", "thesis", "conclusion"]
    text: str


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_id: str
    sections: list[SummarySection]
    source_chunk_ids: list[str] = Field(default_factory=list)


class Term(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    lemma: str
    frequency: int
    category: str
    context: str | None = None
    source_chunk_id: str | None = None


class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    type: Literal["single_choice", "multi_choice", "open_answer"]
    choices: list[str] = Field(default_factory=list)
    answer_idx: int | None = None
    answer_indices: list[int] | None = None
    source_chunk_id: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    authors: str | None = None
    year: int | None = None
    url: str | None = None
    relevance_score: float


class Task(BaseModel):
    """User task accepted from the backend over REST."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str = "demo-user"
    status: TaskStatus = TaskStatus.PLANNING
    requested_outputs: list[Operation]
    conversation_id: str
    documents: list[Document] = Field(default_factory=list)


class FailedOperation(BaseModel):
    op: Operation
    agent: str
    reason: str
    retries: int
    elapsed_sec: float


class ResultStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_sec: float
    agents_called: int
    messages_exchanged: int
    failed_operations: list[FailedOperation] = Field(default_factory=list)


class ResultPayload(BaseModel):
    """Group of four output entities: Summary, Term[], Quiz, Citation[]."""

    model_config = ConfigDict(extra="forbid")

    summary: Summary | None = None
    terms: list[Term] = Field(default_factory=list)
    quiz: list[QuizQuestion] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ResultArtifact(BaseModel):
    """Final artifact (VKR listing 2.5 plus S8 partial-mode extension)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: TaskStatus
    operations: list[Operation]
    result: ResultPayload
    degraded: list[Operation] = Field(default_factory=list)
    stats: ResultStats
