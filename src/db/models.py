"""SQLAlchemy 2.0 models matching VKR ER diagram (figure 2.9)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="demo-user")
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    requested_outputs: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    final_artifact: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    documents: Mapped[list[DocumentRow]] = relationship(back_populates="task", cascade="all, delete-orphan")
    text_chunks: Mapped[list[TextChunkRow]] = relationship(back_populates="task", cascade="all, delete-orphan")


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    document_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)

    task: Mapped[TaskRow] = relationship(back_populates="documents")


class TextChunkRow(Base):
    __tablename__ = "text_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    task: Mapped[TaskRow] = relationship(back_populates="text_chunks")


class SummaryRow(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    introduction: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_points: Mapped[str | None] = mapped_column(Text, nullable=True)
    conclusions: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_chunk_ids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)


class TermRow(Base):
    __tablename__ = "terms"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    text_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("text_chunks.id", ondelete="SET NULL"), nullable=True)
    term: Mapped[str] = mapped_column(Text, nullable=False)
    lemma: Mapped[str | None] = mapped_column(Text, nullable=True)
    frequency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)


class QuizRow(Base):
    __tablename__ = "quizzes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    difficulty: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class CitationRow(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(nullable=True)


Index("idx_text_chunks_task", TextChunkRow.task_id)
Index("idx_terms_task", TermRow.task_id)
