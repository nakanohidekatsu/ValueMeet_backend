# db_control/crud.py
import platform
print("platform", platform.uname())

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, insert, delete, update, select ,BigInteger, Column, and_, or_
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse

from datetime import datetime, date
from typing import List, Optional
import os
from dotenv import load_dotenv
from . import mymodels
# from db_control.connect import engine
from sqlalchemy import func

from .mymodels import Tag
from .mymodels import User, Participant
from pgvector.sqlalchemy import Vector

# 環境変数の読み込み
load_dotenv()

# ●●● nakano
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure環境変数確認
logger.info(f"🔍 DEBUG_HOTSPOTS: {os.getenv('DEBUG_HOTSPOTS')}")


import time
import logging
import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

# Azure App Service 用のログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 環境変数でSQL監視を制御
# DEBUG_SQL = os.getenv("DEBUG_SQL", "false").lower() == "true"
DEBUG_SQL = os.getenv("DEBUG_SQL", "0").lower() in {"1","true","yes","on"}


# SQL実行時間を自動測定（追加コード）
if DEBUG_SQL:
    # SQLクエリの実行時間を測定
    @event.listens_for(Engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault('query_start_time', []).append(time.time())
        
        # クエリの開始ログ
        logger.info(f"🔍 SQL開始: {statement[:100]}...")
    
    @event.listens_for(Engine, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        total = time.time() - conn.info['query_start_time'].pop(-1)
        
        # 実行時間をログ出力
        if total > 0.5:  # 0.5秒以上の場合は警告
            logger.warning(f"🐌 遅いSQL: {total:.4f}秒 - {statement[:100]}...")
        else:
            logger.info(f"⚡ SQL完了: {total:.4f}秒")
        
        # パラメータも表示（デバッグ時）
        if parameters and total > 1.0:  # 1秒以上の場合のみ
            logger.info(f"📝 パラメータ: {parameters}")

# ●●● nakano

# データベース接続設定　#
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/postgres")
# engine = create_engine(DATABASE_URL)

# データベース接続情報
DB_USER = os.getenv('MYSQL_USER')
DB_PASSWORD = os.getenv('MYSQL_PASSWORD')
DB_HOST = os.getenv('MYSQL_SERVER')
DB_PORT = os.getenv('MYSQL_DB_PORT')
DB_NAME = os.getenv('MYSQL_DB')

ssl_cert = str('DigiCertGlobalRootCA.crt.pem')

# MySQLのURL構築
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# データベース接続の設定
def get_database_config():
    """環境変数からデータベース設定を取得"""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL環境変数が設定されていません")
    
    return database_url

# SQLAlchemy設定
# DATABASE_URL = get_database_config() # nakano add ●●●
# engine = create_engine(DATABASE_URL) # nakano add ●●●
# SessionLocal = sessionmaker(autocommit=False, autoflush=False) # nakano add ●●●
# DBエンジン作成は connect.pyに集約
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def get_db_connection():
    """psycopg2用のデータベース接続を取得（修正版）"""
    try:
        # SQLAlchemy形式のDSN文字列をpsycopg2形式に変換
        database_url = get_database_config()
        
        # postgresql+psycopg2:// を postgresql:// に変換
        if database_url.startswith("postgresql+psycopg2://"):
            database_url = database_url.replace("postgresql+psycopg2://", "postgresql://")
        
        # URLを解析
        parsed = urlparse(database_url)
        
        # psycopg2用の接続パラメータを構築
        conn_params = {
            'host': parsed.hostname,
            'port': parsed.port or 5432,
            'database': parsed.path[1:],  # 先頭の'/'を除去
            'user': parsed.username,
            'password': parsed.password
        }
        
        # 接続テスト
        conn = psycopg2.connect(**conn_params)
        return conn
        
    except Exception as e:
        # Fallback: 環境変数から直接接続パラメータを取得
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com"),
                port=int(os.getenv("DB_PORT", "6543")),
                database=os.getenv("DB_NAME", "postgres"),
                user=os.getenv("DB_USER", "postgres.zdzdymwaessgxeojmtpb"),
                password=os.getenv("DB_PASSWORD", "ValueMeet2025"),
                sslmode='require'  # Supabaseは通常SSL必須
            )
            return conn
        except Exception as fallback_error:
            raise Exception(f"データベース接続エラー: {str(fallback_error)}")

# または、環境変数を個別に設定する場合の代替案
def get_db_connection_alternative():
    """環境変数から個別にデータベース接続パラメータを取得する方法"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com"),
            port=os.getenv("DB_PORT", "6543"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres.zdzdymwaessgxeojmtpb"),
            password=os.getenv("DB_PASSWORD", "ValueMeet2025")
        )
        return conn
    except Exception as e:
        raise Exception(f"データベース接続エラー: {str(e)}")
    
def get_db():
    """データベースセッションを取得"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === 認証関連 ===
def get_db_connection():
    """データベース接続を取得"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データベース接続エラー: {str(e)}")

def hash_password(password: str) -> str:
    """パスワードをSHA-256でハッシュ化"""
    return hashlib.sha256(password.encode()).hexdigest()

# === User関連 ===

def get_user_by_id(user_id: str):
    """SQLAlchemyセッションを使用してユーザーを取得"""
    db = SessionLocal()
    try:
        from . import mymodels
        user = db.query(mymodels.User).filter(mymodels.User.user_id == user_id).first()
        return user
    finally:
        db.close()

def search_users_by_name(name: str) -> List[mymodels.User]:
    """名前で部分一致検索"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.User).where(mymodels.User.name.like(f"%{name}%"))
        ).scalars().all()

# === Organization関連 ===

def get_organization_by_id(organization_id: int):
    """SQLAlchemyセッションを使用して組織を取得"""
    if not organization_id:
        return None
    
    db = SessionLocal()
    try:
        from . import mymodels
        org = db.query(mymodels.Organization).filter(mymodels.Organization.organization_id == organization_id).first()
        return org
    finally:
        db.close()

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
    """ユーザーが参加または作成する会議一覧を取得（修正版）"""
    with SessionLocal() as db:
        # サブクエリ1: ユーザーが参加する会議
        participated_meetings = (
            select(mymodels.Meeting.meeting_id)
            .join(mymodels.Participant, mymodels.Meeting.meeting_id == mymodels.Participant.meeting_id)
            .where(mymodels.Participant.user_id == user_id)
        )
        
        # サブクエリ2: ユーザーが作成した会議  
        created_meetings = (
            select(mymodels.Meeting.meeting_id)
            .where(mymodels.Meeting.created_by == user_id)
        )
        
        # メインクエリ: 両方のサブクエリの結果を統合
        query = (
            select(mymodels.Meeting)
            .where(
                or_(
                    mymodels.Meeting.meeting_id.in_(participated_meetings),
                    mymodels.Meeting.meeting_id.in_(created_meetings)
                )
            )
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

def find_meeting_ids_by_tag_vector(db: Session, query_vector: List[float], top_k: int = 5) -> List[int]:
    # 距離 (cosine_distance) の式を一度定義しておく
    distance_expr = mymodels.Tag.vector_embedding.cosine_distance(query_vector)

    # 1) タグごとに最小距離だけを残すサブクエリを作成
    subq = (
        select(
            mymodels.Tag.meeting_id.label("meeting_id"),
            func.min(distance_expr).label("distance")
        )
        .group_by(mymodels.Tag.meeting_id)
        .subquery()
    )

    # 2) ミーティングID を距離でソートして取得
    stmt = (
        select(subq.c.meeting_id)
        .order_by(subq.c.distance)
        .limit(top_k)
    )

    return db.execute(stmt).scalars().all()

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

def get_agenda_by_meeting_id(meeting_id: int):
    """会議IDからアジェンダを取得"""
    db = SessionLocal()
    try:
        agenda = db.query(mymodels.Agenda).filter(
            mymodels.Agenda.meeting_id == meeting_id
        ).first()
        return agenda
    finally:
        db.close()

def get_users_by_meeting_ids(
    db: Session,
    meeting_ids: list[int]
) -> list[User]:
    """
    与えられた meeting_id リストに参加しているユーザーを取得。
    同一ユーザーが重複して返らないよう DISTINCT にしています。
    """
    stmt = (
        select(User)
        .join(Participant, Participant.user_id == User.user_id)
        .where(Participant.meeting_id.in_(meeting_ids))
        .distinct()
    )
    return db.execute(stmt).scalars().all()

def get_meetings_by_user_with_details(user_id: str, start_datetime: Optional[str] = None,
                        end_datetime: Optional[str] = None, organization_id: Optional[int] = None,
                        meeting_type: Optional[str] = None):
    """ユーザーが参加または作成した会議一覧を詳細情報と共に取得（パフォーマンス最適化版）"""
    with SessionLocal() as db:
        # 基本クエリ：会議、主催者、主催者組織、参加者役割を一度に取得
        query = (
            select(
                mymodels.Meeting,
                mymodels.User.name.label('creator_name'),
                mymodels.Organization.organization_name.label('creator_organization_name'),
                mymodels.Participant.role_type
            )
            .outerjoin(mymodels.User, mymodels.Meeting.created_by == mymodels.User.user_id)
            .outerjoin(mymodels.Organization, mymodels.User.organization_id == mymodels.Organization.organization_id)
            .outerjoin(
                mymodels.Participant, 
                and_(
                    mymodels.Meeting.meeting_id == mymodels.Participant.meeting_id,
                    mymodels.Participant.user_id == user_id
                )
            )
            .where(
                or_(
                    mymodels.Meeting.created_by == user_id,  # ユーザーが作成した会議
                    mymodels.Participant.user_id == user_id  # ユーザーが参加する会議
                )
            )
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
            query = query.where(mymodels.User.organization_id == organization_id)
        
        # 会議タイプフィルタ
        if meeting_type:
            query = query.where(mymodels.Meeting.meeting_type == meeting_type)
        
        # 日時順でソート
        query = query.order_by(mymodels.Meeting.date_time)
        
        return db.execute(query).all()

def get_users_by_organization(organization_id: int) -> List[mymodels.User]:
    """組織に所属するユーザー一覧を取得"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.User).where(mymodels.User.organization_id == organization_id)
        ).scalars().all()

def get_meetings_by_organization_with_details(
    organization_id: int, 
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None, 
    meeting_type: Optional[str] = None
):
    """組織の会議一覧を詳細情報と共に取得"""
    with SessionLocal() as db:
        # 組織に所属するユーザーが作成または参加した会議を取得
        query = (
            select(
                mymodels.Meeting,
                mymodels.User.name.label('creator_name'),
                mymodels.Organization.organization_name.label('creator_organization_name'),
                mymodels.Participant.role_type
            )
            .outerjoin(mymodels.User, mymodels.Meeting.created_by == mymodels.User.user_id)
            .outerjoin(mymodels.Organization, mymodels.User.organization_id == mymodels.Organization.organization_id)
            .outerjoin(mymodels.Participant, mymodels.Meeting.meeting_id == mymodels.Participant.meeting_id)
            .where(
                or_(
                    # 組織のメンバーが作成した会議
                    mymodels.User.organization_id == organization_id,
                    # 組織のメンバーが参加している会議
                    mymodels.Participant.user_id.in_(
                        select(mymodels.User.user_id).where(mymodels.User.organization_id == organization_id)
                    )
                )
            )
        )
        
        # 日時フィルタ
        if start_datetime:
            start_dt = datetime.strptime(start_datetime, "%Y/%m/%d %H:%M")
            query = query.where(mymodels.Meeting.date_time >= start_dt)
        
        if end_datetime:
            end_dt = datetime.strptime(end_datetime, "%Y/%m/%d %H:%M")
            query = query.where(mymodels.Meeting.date_time <= end_dt)
        
        # 会議タイプフィルタ
        if meeting_type:
            query = query.where(mymodels.Meeting.meeting_type == meeting_type)
        
        # 日時順でソート
        query = query.order_by(mymodels.Meeting.date_time)
        
        return db.execute(query).all()

def get_participant_count(db: Session, meeting_id: int) -> int:
    """指定された会議の参加者数を取得"""
    try:
        count = db.execute(
            select(func.count(mymodels.Participant.participant_id))
            .where(mymodels.Participant.meeting_id == meeting_id)
        ).scalar() or 0
        return count
    except Exception as e:
        print(f"参加者数取得エラー: {e}")
        return 0

def get_participant_details_by_meeting_id(meeting_id: int, db: Session):
    """指定された会議の参加者詳細を取得"""
    try:
        query = """
            SELECT p.user_id, u.name, o.organization_name, p.role_type
            FROM participants p
            LEFT JOIN users u ON p.user_id = u.user_id
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE p.meeting_id = :meeting_id
            ORDER BY u.name
        """
        result = db.execute(text(query), {"meeting_id": meeting_id})
        return result.fetchall()
    except Exception as e:
        print(f"参加者詳細取得エラー: {e}")
        return []
    
def get_meeting_by_id(meeting_id: int) -> Optional[mymodels.Meeting]:
    """会議IDで会議情報を取得"""
    with SessionLocal() as db:
        return db.execute(
            select(mymodels.Meeting).where(mymodels.Meeting.meeting_id == meeting_id)
        ).scalar_one_or_none()

def delete_meeting_and_related_data(meeting_id: int):
    """会議とその関連データを削除"""
    with SessionLocal() as db:
        try:
            # トランザクション開始
            
            # 関連するタグを削除
            db.execute(
                delete(mymodels.Tag).where(mymodels.Tag.meeting_id == meeting_id)
            )
            
            # 関連する参加者を削除
            db.execute(
                delete(mymodels.Participant).where(mymodels.Participant.meeting_id == meeting_id)
            )
            
            # 関連するアジェンダを削除
            db.execute(
                delete(mymodels.Agenda).where(mymodels.Agenda.meeting_id == meeting_id)
            )
            
            # ファイル添付があれば削除
            db.execute(
                delete(mymodels.FileAttachment).where(mymodels.FileAttachment.meeting_id == meeting_id)
            )
            
            # 会議評価があれば削除
            db.execute(
                delete(mymodels.MeetingEvaluation).where(mymodels.MeetingEvaluation.meeting_id == meeting_id)
            )
            
            # TODOがあれば削除
            db.execute(
                delete(mymodels.Todo).where(mymodels.Todo.meeting_id == meeting_id)
            )
            
            # 議事録があれば削除
            db.execute(
                delete(mymodels.Transcript).where(mymodels.Transcript.meeting_id == meeting_id)
            )
            
            # 最後に会議本体を削除
            db.execute(
                delete(mymodels.Meeting).where(mymodels.Meeting.meeting_id == meeting_id)
            )
            
            # コミット
            db.commit()
            
        except Exception as e:
            # エラー時はロールバック
            db.rollback()
            raise e

def get_participant_count(db: Session, meeting_id: int) -> int:
    """指定された会議の参加者数を取得"""
    try:
        from sqlalchemy import select, func
        count = db.execute(
            select(func.count(mymodels.Participant.participant_id))
            .where(mymodels.Participant.meeting_id == meeting_id)
        ).scalar()
        return count if count is not None else 0
    except Exception as e:
        print(f"参加者数取得エラー: {e}")

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

# テスト用のデータベース接続確認
def test_db_connection():
    """データベース接続をテストする"""
    try:
        # SQLAlchemy接続テスト
        db = SessionLocal()
        result = db.execute(text("SELECT 1"))
        db.close()
        print("✅ SQLAlchemy接続成功")
        
        # psycopg2接続テスト
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
        conn.close()
        print("✅ psycopg2接続成功")
        
        return True
    except Exception as e:
        print(f"❌ データベース接続エラー: {e}")
        return False

if __name__ == "__main__":
    test_db_connection()
    

    
