# mymodels.py
# mymodels.py

from sqlalchemy import (
    Integer, String, Text, DateTime, Date,
    ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector  # PostgreSQL pgvector extension
from sqlalchemy import Identity
from typing import Optional

class Base(DeclarativeBase):
    pass

class Organization(Base):
    __tablename__ = 'organizations'

    organization_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )
    organization_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

class User(Base):
    __tablename__ = 'users'

    user_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True
    )
    # 認証機能のためにpasswordフィールドを追加
    password: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    organization_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('organizations.organization_id'),
        nullable=True
    )

class Meeting(Base):
    __tablename__ = 'meetings'

    meeting_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )
    meeting_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    meeting_mode: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )
    date_time: Mapped[DateTime] = mapped_column(
        DateTime,
        nullable=False
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=True
    )
    # 一時保存機能のためのstatusフィールドを追加
    status: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        default='scheduled'  # 'scheduled', 'draft', 'completed'
    )

class Agenda(Base):
    __tablename__ = 'agendas'

    agenda_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    purpose: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    topic: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

class Tag(Base):
    __tablename__ = 'tags'

    # 自動採番を有効化
    tag_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    tag: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    vector_embedding: Mapped[list[float]] = mapped_column(
        Vector(1536),
        nullable=False
    )
    
class Participant(Base):
    __tablename__ = 'participants'

    participant_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=True
    )
    role_type: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True
    )

class Todo(Base):
    __tablename__ = 'todos'

    todo_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=True
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    status: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )
    due_date: Mapped[Optional[Date]] = mapped_column(
        Date,
        nullable=True
    )

class Reminder(Base):
    __tablename__ = 'reminders'

    reminder_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=True
    )
    todo_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('todos.todo_id'),
        nullable=True
    )
    remind_at: Mapped[Optional[DateTime]] = mapped_column(
        DateTime,
        nullable=True
    )

class FileAttachment(Base):
    __tablename__ = 'file_attachments'

    file_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    filename: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    file_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

class Transcript(Base):
    __tablename__ = 'transcripts'

    transcript_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=True
    )
    speaker_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=True
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    timestamp: Mapped[Optional[DateTime]] = mapped_column(
        DateTime,
        nullable=True
    )

# 新規追加：会議評価テーブル
class MeetingEvaluation(Base):
    __tablename__ = 'meeting_evaluations'

    evaluation_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    meeting_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('meetings.meeting_id'),
        nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey('users.user_id'),
        nullable=False
    )
    satisfaction: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # 1-5の評価
    )
    rejoin_intent: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # 1-5の評価
    )
    self_contribution: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # 1-5の評価
    )
    facilitator_rating: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # 1-5の評価
    )
    comment: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    created_at: Mapped[Optional[DateTime]] = mapped_column(
        DateTime,
        nullable=True,
        default=lambda: __import__('datetime').datetime.utcnow()
    )
    