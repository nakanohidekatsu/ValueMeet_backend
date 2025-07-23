# uname() error回避
import platform
print("platform", platform.uname())

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, insert, delete, update, select ,BigInteger, Column, and_, or_
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session
import json
import pandas as pd

from datetime import datetime, date
from typing import List, Optional
import os
from dotenv import load_dotenv
from . import mymodels
from db_control.connect import engine
from sqlalchemy import func

# 環境変数の読み込み
load_dotenv()

# データベース接続設定　#
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/meeting_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """データベースセッションを取得"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === User関連 ===

def get_user_by_id(user_id: str) -> Optional[mymodels.User]:
    """ユーザーIDでユーザー情報を取得"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.User).where(mymodels.User.user_id == user_id)
        ).scalar_one_or_none()

def search_users_by_name(name: str) -> List[mymodels.User]:
    """名前で部分一致検索"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.User).where(mymodels.User.name.like(f"%{name}%"))
        ).scalars().all()

# === Organization関連 ===

def get_organization_by_id(organization_id: int) -> Optional[mymodels.Organization]:
    """組織IDで組織情報を取得"""
    if organization_id is None:
        return None
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.Organization).where(
                mymodels.Organization.organization_id == organization_id
            )
        ).scalar_one_or_none()

# === Meeting関連 ===

def create_meeting(title: str, meeting_type: Optional[str], meeting_mode: Optional[str], 
                  date_time: datetime, created_by: str) -> int:
    """会議を作成"""
    with SessionLocal() as db:
        meeting = mymodels.Meeting(
            title=title,
            meeting_type=meeting_type,
            meeting_mode=meeting_mode,
            date_time=date_time,
            created_by=created_by
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
        return meeting.meeting_id

def get_meetings_by_user(user_id: str, start_datetime: Optional[str] = None,
                        end_datetime: Optional[str] = None, organization_id: Optional[int] = None,
                        meeting_type: Optional[str] = None) -> List[mymodels.Meeting]:
    """ユーザーが参加する会議一覧を取得"""
    with SessionLocal() as db:
        # 基本クエリ：参加者テーブルとJOIN
        query = (
            select(mymodels.Meeting)
            .join(mymodels.Participant, mymodels.Meeting.meeting_id == mymodels.Participant.meeting_id)
            .where(mymodels.Participant.user_id == user_id)
        )
        
        # 日時フィルタ
        if start_datetime:
            start_dt = datetime.strptime(start_datetime, "%Y/%m/%d %H:%M")
            query = query.where(mymodels.Meeting.date_time >= start_dt)
        
        if end_datetime:
            end_dt = datetime.strptime(end_datetime, "%Y/%m/%d %H:%M")
            query = query.where(mymodels.Meeting.date_time <= end_dt)
        
        # 組織フィルタ（主催者の組織）
        if organization_id:
            query = (
                query.join(mymodels.User, mymodels.Meeting.created_by == mymodels.User.user_id)
                .where(mymodels.User.organization_id == organization_id)
            )
        
        # 会議タイプフィルタ
        if meeting_type:
            query = query.where(mymodels.Meeting.meeting_type == meeting_type)
        
        # 日時順でソート
        query = query.order_by(mymodels.Meeting.date_time)
        
        return db.execute(query).scalars().all()

# === Agenda関連 ===

def create_agenda(meeting_id: int, purpose: Optional[str], topic: Optional[str]) -> int:
    """アジェンダを作成"""
    with SessionLocal() as db:
        agenda = mymodels.Agenda(
            meeting_id=meeting_id,
            purpose=purpose,
            topic=topic
        )
        db.add(agenda)
        db.commit()
        db.refresh(agenda)
        return agenda.agenda_id

# === Tag関連 ===

def create_tag(meeting_id: int, tag: str, vector_embedding: List[float]) -> int:
    """タグを作成"""
    with SessionLocal() as db:
        tag_obj = mymodels.Tag(
            meeting_id=meeting_id,
            tag=tag,
            vector_embedding=vector_embedding
        )
        db.add(tag_obj)
        db.commit()
        db.refresh(tag_obj)
        return tag_obj.tag_id

def generate_tags_from_topic(topic: str) -> List[str]:
    """トピックからタグを生成（ChatGPT API呼び出し想定）"""
    # 実際の実装では、OpenAI APIを呼び出してタグを生成
    # ここではダミーの実装
    import re
    
    # 簡単なキーワード抽出（実際にはChatGPT APIを使用）
    words = re.findall(r'\w+', topic)
    # 重要そうなキーワードを抽出（実際にはより高度な処理）
    keywords = [word for word in words if len(word) > 2][:3]
    
    return keywords if keywords else ["一般", "会議", "議論"]

def get_recommended_users_by_tag(tag: str) -> List[mymodels.User]:
    """タグに基づいておすすめユーザーを取得"""
    with SessionLocal() as db:
        # 実際の実装では、ベクトル類似度検索を行う
        # ここではダミーの実装：同じタグに関連した会議に参加したユーザー
        subquery = (
            select(mymodels.Participant.user_id)
            .join(mymodels.Tag, mymodels.Participant.meeting_id == mymodels.Tag.meeting_id)
            .where(mymodels.Tag.tag.like(f"%{tag}%"))
            .distinct()
        )
        
        return db.execute(
            select(mymodels.User).where(mymodels.User.user_id.in_(subquery)).limit(5)
        ).scalars().all()

# === Participant関連 ===

def create_participant(meeting_id: int, user_id: str, role_type: Optional[str]) -> int:
    """参加者を作成"""
    with SessionLocal() as db:
        participant = mymodels.Participant(
            meeting_id=meeting_id,
            user_id=user_id,
            role_type=role_type
        )
        db.add(participant)
        db.commit()
        db.refresh(participant)
        return participant.participant_id

def get_participant_role(meeting_id: int, user_id: str) -> Optional[mymodels.Participant]:
    """特定会議での参加者の役割を取得"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.Participant).where(
                and_(
                    mymodels.Participant.meeting_id == meeting_id,
                    mymodels.Participant.user_id == user_id
                )
            )
        ).scalar_one_or_none()

# === データベース初期化用の関数 ===

def create_sample_data():
    """サンプルデータを作成"""
    with SessionLocal() as db:
        # 組織データ
        if not db.execute(select(mymodels.Organization)).first():
            organizations = [
                mymodels.Organization(organization_id=1, organization_name="開発部"),
                mymodels.Organization(organization_id=2, organization_name="営業部"),
                mymodels.Organization(organization_id=3, organization_name="マーケティング部"),
            ]
            db.add_all(organizations)
            db.commit()
        
        # ユーザーデータ
        if not db.execute(select(mymodels.User)).first():
            users = [
                mymodels.User(user_id="user001", name="田中太郎", email="tanaka@example.com", organization_id=1),
                mymodels.User(user_id="user002", name="佐藤花子", email="sato@example.com", organization_id=2),
                mymodels.User(user_id="user003", name="鈴木一郎", email="suzuki@example.com", organization_id=3),
            ]
            db.add_all(users)
            db.commit()

if __name__ == "__main__":
    # テスト用
    create_sample_data()
    print("Sample data created successfully")
    
