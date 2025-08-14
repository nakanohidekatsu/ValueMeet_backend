# app.py - 最適化版バックエンド

from fastapi import FastAPI, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time as datetime_time
from sqlalchemy import create_engine, text, select, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
import asyncio
from contextlib import asynccontextmanager
import hashlib

from db_control import crud, mymodels
from db_control.create_tables import init_db
from db_control.crud import SessionLocal

# schemas.pyから必要なモデルをimport
from schemas import LoginRequest, LoginResponse, ResetPasswordRequest, MessageResponse

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 環境変数とデータベース設定
load_dotenv()

def get_database_url():
    """環境変数からデータベースURLを取得（最適化版）"""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    host = os.getenv("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com")
    port = os.getenv("DB_PORT", "6543")
    database = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres.zdzdymwaessgxeojmtpb")
    password = os.getenv("DB_PASSWORD", "ValueMeet2025")
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

# SQLAlchemy設定（コネクションプール最適化）
try:
    DATABASE_URL = get_database_url()
    logger.info(f"使用するDATABASE_URL: {DATABASE_URL}")
    
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,  # コネクションプール数
        max_overflow=20,  # 最大オーバーフロー
        pool_pre_ping=True,
        pool_recycle=3600,  # 1時間でコネクション再作成
        echo=False,
        connect_args={
            "connect_timeout": 10,
            "command_timeout": 30
        }
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("✅ SQLAlchemy設定完了")
    
except Exception as e:
    logger.error(f"❌ データベース設定エラー: {e}")
    raise e

# Redis設定（キャッシュ用）
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    # Redis接続テスト
    redis_client.ping()
    logger.info("✅ Redis接続成功")
    REDIS_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Redis接続失敗: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

# OpenAIクライアント
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI APIクライアント初期化完了")
else:
    logger.warning("⚠️ OPENAI_API_KEY が設定されていません")

# === Pydantic モデル（最適化版） ===

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    organization_id: int
    organization_name: str

class DepartmentMember(BaseModel):
    user_id: str
    name: str
    organization_name: str

class MeetingListItemOptimized(BaseModel):
    """最適化された会議リスト項目"""
    meeting_id: int
    title: str
    meeting_type: Optional[str] = None
    meeting_mode: Optional[str] = None
    date_time: str
    end_time: Optional[str] = None
    name: str = ""  # 作成者名
    organization_name: str = ""  # 作成者組織名
    role_type: Optional[str] = None
    purpose: Optional[str] = None
    status: Optional[str] = "scheduled"
    participants: int = 0
    rule_violation: Optional[bool] = False
    created_by: Optional[str] = None
    agenda: Optional[List[str]] = None
    meeting_cost: Optional[float] = None  # 会議コスト

class MeetingBatchRequest(BaseModel):
    """一括取得リクエスト"""
    user_id: str
    filter_type: str = "my"  # my, department, member
    selected_member: Optional[str] = None
    search_query: Optional[str] = ""
    meeting_type: Optional[str] = None
    period_type: Optional[str] = "week"
    current_date: Optional[str] = None
    limit: Optional[int] = 1000

class PaginatedResponse(BaseModel):
    """ページネーションレスポンス"""
    meetings: List[MeetingListItemOptimized]
    pagination: Dict[str, Any]
    cache_info: Optional[Dict[str, Any]] = None

class PerformanceMetrics(BaseModel):
    """パフォーマンス測定結果"""
    endpoint: str
    execution_time: float
    query_count: int
    cache_hit: bool
    timestamp: str

# === ユーティリティ関数 ===

def generate_cache_key(*args) -> str:
    """キャッシュキー生成"""
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()

def get_from_cache(key: str) -> Optional[Any]:
    """キャッシュから取得"""
    if not REDIS_AVAILABLE:
        return None
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.warning(f"キャッシュ取得エラー: {e}")
        return None

def set_to_cache(key: str, data: Any, ttl: int = 300) -> bool:
    """キャッシュに保存"""
    if not REDIS_AVAILABLE:
        return False
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
        return True
    except Exception as e:
        logger.warning(f"キャッシュ保存エラー: {e}")
        return False

def calculate_meeting_cost(participants: int, start_time: str, end_time: str) -> float:
    """会議コスト計算"""
    try:
        if not start_time or not end_time or participants == 0:
            return 0.0
        
        start = datetime.strptime(f"2000-01-01 {start_time}", "%Y-%m-%d %H:%M")
        end = datetime.strptime(f"2000-01-01 {end_time}", "%Y-%m-%d %H:%M")
        duration_hours = (end - start).total_seconds() / 3600
        
        if duration_hours <= 0:
            return 0.0
        
        return participants * 5000 * duration_hours
    except Exception:
        return 0.0

# === パフォーマンス監視デコレータ ===

def monitor_performance(endpoint_name: str):
    """パフォーマンス監視デコレータ"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # パフォーマンスログ
                logger.info(f"[PERF] {endpoint_name}: {execution_time:.3f}s")
                
                # メトリクス保存（Redis利用可能時）
                if REDIS_AVAILABLE:
                    metrics = PerformanceMetrics(
                        endpoint=endpoint_name,
                        execution_time=execution_time,
                        query_count=1,  # 実際のクエリ数を計測する場合は拡張
                        cache_hit=False,  # キャッシュヒット情報
                        timestamp=datetime.now().isoformat()
                    )
                    redis_client.lpush(
                        "performance_metrics", 
                        metrics.json()
                    )
                    redis_client.ltrim("performance_metrics", 0, 1000)  # 最新1000件保持
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[ERROR] {endpoint_name}: {execution_time:.3f}s - {str(e)}")
                raise
        return wrapper
    return decorator

# === データベース最適化関数 ===

async def ensure_database_indexes(db: Session):
    """必要なインデックスを確実に作成"""
    indexes = [
        # 会議テーブル
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meetings_created_by_date ON meetings(created_by, date_time DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meetings_date_time ON meetings(date_time DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meetings_status ON meetings(status)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meetings_type ON meetings(meeting_type)",
        
        # 参加者テーブル
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_participants_meeting_id ON participants(meeting_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_participants_user_id ON participants(user_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_participants_user_meeting ON participants(user_id, meeting_id)",
        
        # ユーザーテーブル
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_organization ON users(organization_id)",
        
        # アジェンダテーブル
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agendas_meeting_id ON agendas(meeting_id)",
        
        # 複合インデックス
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meetings_creator_org_date ON meetings(created_by) INCLUDE (date_time, status, meeting_type)"
    ]
    
    for index_sql in indexes:
        try:
            await asyncio.to_thread(db.execute, text(index_sql))
            await asyncio.to_thread(db.commit)
            logger.info(f"✅ インデックス確認/作成: {index_sql.split()[-1]}")
        except Exception as e:
            logger.warning(f"⚠️ インデックス作成スキップ: {e}")
            await asyncio.to_thread(db.rollback)

# === アプリケーション初期化 ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時
    logger.info("🚀 アプリケーション開始")
    
    # データベース初期化
    try:
        init_db()
        logger.info("✅ データベーステーブル初期化完了")
    except Exception as e:
        logger.error(f"❌ データベース初期化エラー: {e}")
    
    # インデックス作成
    try:
        db = SessionLocal()
        await ensure_database_indexes(db)
        db.close()
        logger.info("✅ データベースインデックス確認完了")
    except Exception as e:
        logger.error(f"❌ インデックス作成エラー: {e}")
    
    # キャッシュクリア
    if REDIS_AVAILABLE:
        try:
            redis_client.flushdb()
            logger.info("✅ キャッシュクリア完了")
        except Exception as e:
            logger.warning(f"⚠️ キャッシュクリア失敗: {e}")
    
    yield
    
    # 終了時
    logger.info("⏹️ アプリケーション終了")

# FastAPIアプリケーション作成
app = FastAPI(
    title="Meeting Management API - Optimized",
    version="2.0.0",
    description="高パフォーマンス会議管理API",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === データベースセッション ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === 認証エンドポイント（既存と同じ） ===

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """ログイン認証"""
    try:
        query = text("""
            SELECT u.user_id, u.name, u.email, u.organization_id, u.password, 
                   COALESCE(o.organization_name, '') as organization_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE u.user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": request.user_id})
        user = result.fetchone()
        
        if not user or user.password != request.password:
            raise HTTPException(status_code=401, detail="認証に失敗しました")
        
        return LoginResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            organization_id=user.organization_id or 0,
            organization_name=user.organization_name or ""
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ログインエラー: {e}")
        raise HTTPException(status_code=500, detail="ログイン処理中にエラーが発生しました")

# === 最適化された会議管理エンドポイント ===

@app.post("/meeting_list_optimized", response_model=List[MeetingListItemOptimized])
@monitor_performance("meeting_list_optimized")
async def get_meeting_list_optimized(
    request: MeetingBatchRequest,
    db: Session = Depends(get_db)
):
    """
    最適化された会議一覧取得API
    - 単一クエリで全データ取得
    - N+1問題解決
    - キャッシュ機能付き
    """
    try:
        # キャッシュキー生成
        cache_key = f"meetings_opt:{generate_cache_key(request.dict())}"
        
        # キャッシュチェック
        cached_data = get_from_cache(cache_key)
        if cached_data:
            logger.info(f"キャッシュヒット: {cache_key}")
            return [MeetingListItemOptimized(**item) for item in cached_data]
        
        # 最適化されたクエリ（単一クエリで全データ取得）
        base_query = text("""
            WITH meeting_participants AS (
                SELECT 
                    meeting_id,
                    COUNT(*) as participant_count
                FROM participants 
                GROUP BY meeting_id
            ),
            user_meetings AS (
                SELECT DISTINCT m.meeting_id
                FROM meetings m
                LEFT JOIN participants p ON m.meeting_id = p.meeting_id
                WHERE 
                    CASE 
                        WHEN :filter_type = 'my' THEN 
                            (m.created_by = :user_id OR p.user_id = :user_id)
                        WHEN :filter_type = 'department' THEN 
                            EXISTS (
                                SELECT 1 FROM users creator 
                                WHERE creator.user_id = m.created_by 
                                AND creator.organization_id = (
                                    SELECT organization_id FROM users WHERE user_id = :user_id
                                )
                            )
                        WHEN :filter_type = 'member' AND :selected_member IS NOT NULL THEN 
                            (m.created_by = :selected_member OR p.user_id = :selected_member)
                        ELSE TRUE
                    END
            )
            SELECT 
                m.meeting_id,
                m.title,
                m.meeting_type,
                m.meeting_mode,
                m.date_time,
                m.end_time,
                m.status,
                m.rule_violation,
                m.created_by,
                creator.name as creator_name,
                creator_org.organization_name,
                COALESCE(mp.participant_count, 0) as participant_count,
                a.purpose,
                CASE 
                    WHEN a.topic IS NOT NULL THEN 
                        ARRAY[a.topic]
                    ELSE 
                        ARRAY[]::text[]
                END as agenda_topics
            FROM meetings m
            INNER JOIN user_meetings um ON m.meeting_id = um.meeting_id
            LEFT JOIN users creator ON m.created_by = creator.user_id
            LEFT JOIN organizations creator_org ON creator.organization_id = creator_org.organization_id
            LEFT JOIN meeting_participants mp ON m.meeting_id = mp.meeting_id
            LEFT JOIN agendas a ON m.meeting_id = a.meeting_id
            WHERE 
                (:search_query = '' OR 
                 m.title ILIKE :search_pattern OR 
                 a.purpose ILIKE :search_pattern OR
                 creator.name ILIKE :search_pattern)
                AND (:meeting_type IS NULL OR m.meeting_type = :meeting_type)
            ORDER BY m.date_time DESC
            LIMIT :limit
        """)
        
        params = {
            'user_id': request.user_id,
            'filter_type': request.filter_type,
            'selected_member': request.selected_member,
            'search_query': request.search_query or '',
            'search_pattern': f'%{request.search_query or ""}%',
            'meeting_type': request.meeting_type if request.meeting_type != 'all' else None,
            'limit': min(request.limit or 1000, 1000)
        }
        
        # クエリ実行
        start_time = time.time()
        result = db.execute(base_query, params)
        meetings = result.fetchall()
        query_time = time.time() - start_time
        
        logger.info(f"クエリ実行時間: {query_time:.3f}s, 取得件数: {len(meetings)}")
        
        # レスポンスデータ構築
        response_data = []
        for meeting in meetings:
            # 会議コスト計算
            meeting_cost = 0.0
            if meeting.end_time:
                start_time_str = meeting.date_time.strftime("%H:%M")
                end_time_str = meeting.end_time.strftime("%H:%M")
                meeting_cost = calculate_meeting_cost(
                    meeting.participant_count + 1,  # +1 for facilitator
                    start_time_str,
                    end_time_str
                )
            
            meeting_data = MeetingListItemOptimized(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                end_time=meeting.end_time.strftime("%H:%M") if meeting.end_time else None,
                name=meeting.creator_name or "",
                organization_name=meeting.organization_name or "",
                role_type=None,  # 必要に応じて後で拡張
                purpose=meeting.purpose,
                status=meeting.status or "scheduled",
                participants=meeting.participant_count or 0,
                rule_violation=meeting.rule_violation or False,
                created_by=meeting.created_by,
                agenda=meeting.agenda_topics or [],
                meeting_cost=meeting_cost
            )
            response_data.append(meeting_data)
        
        # キャッシュに保存（5分間）
        cache_data = [item.dict() for item in response_data]
        set_to_cache(cache_key, cache_data, ttl=300)
        
        logger.info(f"会議一覧取得完了: {len(response_data)}件")
        return response_data
        
    except Exception as e:
        logger.error(f"最適化会議一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"会議一覧取得に失敗しました: {str(e)}")

@app.get("/meeting_list_paginated", response_model=PaginatedResponse)
@monitor_performance("meeting_list_paginated")
async def get_meeting_list_paginated(
    user_id: str = Query(...),
    page: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    filter_type: str = Query("my"),
    search_query: Optional[str] = Query(""),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """ページネーション対応の会議一覧API"""
    try:
        offset = page * limit
        
        # キャッシュキー
        cache_key = f"meetings_page:{generate_cache_key(user_id, page, limit, filter_type, search_query, meeting_type)}"
        cached_data = get_from_cache(cache_key)
        
        if cached_data:
            return PaginatedResponse(**cached_data, cache_info={"hit": True, "key": cache_key})
        
        # 件数取得クエリ
        count_query = text("""
            SELECT COUNT(DISTINCT m.meeting_id)
            FROM meetings m
            LEFT JOIN users creator ON m.created_by = creator.user_id
            LEFT JOIN participants p ON m.meeting_id = p.meeting_id
            LEFT JOIN agendas a ON m.meeting_id = a.meeting_id
            WHERE (m.created_by = :user_id OR p.user_id = :user_id)
            AND (:search_query = '' OR 
                 m.title ILIKE :search_pattern OR 
                 a.purpose ILIKE :search_pattern)
            AND (:meeting_type IS NULL OR m.meeting_type = :meeting_type)
        """)
        
        # データ取得クエリ（最適化版）
        data_query = text("""
            WITH meeting_data AS (
                SELECT DISTINCT
                    m.meeting_id,
                    m.title,
                    m.meeting_type,
                    m.meeting_mode,
                    m.date_time,
                    m.end_time,
                    m.status,
                    m.rule_violation,
                    m.created_by,
                    creator.name as creator_name,
                    creator_org.organization_name,
                    (SELECT COUNT(*) FROM participants p2 WHERE p2.meeting_id = m.meeting_id) as participant_count,
                    a.purpose
                FROM meetings m
                LEFT JOIN users creator ON m.created_by = creator.user_id
                LEFT JOIN organizations creator_org ON creator.organization_id = creator_org.organization_id
                LEFT JOIN participants p ON m.meeting_id = p.meeting_id
                LEFT JOIN agendas a ON m.meeting_id = a.meeting_id
                WHERE (m.created_by = :user_id OR p.user_id = :user_id)
                AND (:search_query = '' OR 
                     m.title ILIKE :search_pattern OR 
                     a.purpose ILIKE :search_pattern)
                AND (:meeting_type IS NULL OR m.meeting_type = :meeting_type)
            )
            SELECT * FROM meeting_data
            ORDER BY date_time DESC
            LIMIT :limit OFFSET :offset
        """)
        
        params = {
            'user_id': user_id,
            'search_query': search_query or '',
            'search_pattern': f'%{search_query or ""}%',
            'meeting_type': meeting_type if meeting_type != 'all' else None,
            'limit': limit,
            'offset': offset
        }
        
        # 並列実行
        total_count = db.execute(count_query, params).scalar()
        meetings = db.execute(data_query, params).fetchall()
        
        # レスポンス構築
        meeting_list = []
        for m in meetings:
            meeting_cost = 0.0
            if m.end_time:
                start_time_str = m.date_time.strftime("%H:%M")
                end_time_str = m.end_time.strftime("%H:%M")
                meeting_cost = calculate_meeting_cost(
                    m.participant_count + 1,
                    start_time_str,
                    end_time_str
                )
            
            meeting_list.append(MeetingListItemOptimized(
                meeting_id=m.meeting_id,
                title=m.title,
                meeting_type=m.meeting_type,
                meeting_mode=m.meeting_mode,
                date_time=m.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                end_time=m.end_time.strftime("%H:%M") if m.end_time else None,
                name=m.creator_name or "",
                organization_name=m.organization_name or "",
                purpose=m.purpose,
                status=m.status or "scheduled",
                participants=m.participant_count or 0,
                rule_violation=m.rule_violation or False,
                created_by=m.created_by,
                meeting_cost=meeting_cost
            ))
        
        pagination_info = {
            "page": page,
            "limit": limit,
            "total": total_count,
            "has_next": (offset + limit) < total_count,
            "has_prev": page > 0,
            "total_pages": (total_count + limit - 1) // limit
        }
        
        response = PaginatedResponse(
            meetings=meeting_list,
            pagination=pagination_info,
            cache_info={"hit": False, "key": cache_key}
        )
        
        # キャッシュ保存
        set_to_cache(cache_key, response.dict(), ttl=180)  # 3分
        
        return response
        
    except Exception as e:
        logger.error(f"ページネーション取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"ページネーション取得に失敗しました: {str(e)}")

# === 既存API（互換性維持＋最適化） ===

@app.get("/meeting_list", response_model=List[MeetingListItemOptimized])
async def get_meeting_list_legacy(
    user_id: str = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    organization_id: Optional[int] = Query(None),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """既存API（互換性維持＋最適化）"""
    # 新しい最適化APIに転送
    request = MeetingBatchRequest(
        user_id=user_id,
        filter_type="my",
        meeting_type=meeting_type
    )
    return await get_meeting_list_optimized(request, db)

@app.get("/department_meetings", response_model=List[MeetingListItemOptimized])
async def get_department_meetings_optimized(
    organization_id: int = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """部内会議一覧取得API（最適化版）"""
    try:
        # 組織に所属するユーザーを1つ取得（フィルタリング用）
        user_query = text("SELECT user_id FROM users WHERE organization_id = :org_id LIMIT 1")
        user_result = db.execute(user_query, {"org_id": organization_id})
        user = user_result.fetchone()
        
        if not user:
            return []
        
        request = MeetingBatchRequest(
            user_id=user.user_id,
            filter_type="department",
            meeting_type=meeting_type
        )
        return await get_meeting_list_optimized(request, db)
        
    except Exception as e:
        logger.error(f"部内会議取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/member_meetings", response_model=List[MeetingListItemOptimized])
async def get_member_meetings_optimized(
    member_id: str = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """担当者会議一覧取得API（最適化版）"""
    request = MeetingBatchRequest(
        user_id=member_id,
        filter_type="my",
        meeting_type=meeting_type
    )
    return await get_meeting_list_optimized(request, db)

@app.get("/department_members", response_model=List[DepartmentMember])
async def get_department_members_optimized(
    organization_id: int = Query(...),
    db: Session = Depends(get_db)
):
    """部内メンバー一覧取得API（キャッシュ対応）"""
    try:
        cache_key = f"dept_members:{organization_id}"
        cached_data = get_from_cache(cache_key)
        
        if cached_data:
            return [DepartmentMember(**item) for item in cached_data]
        
        query = text("""
            SELECT u.user_id, u.name, o.organization_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE u.organization_id = :organization_id
            ORDER BY u.name
        """)
        
        result = db.execute(query, {"organization_id": organization_id})
        members = result.fetchall()
        
        response_data = [
            DepartmentMember(
                user_id=member.user_id,
                name=member.name,
                organization_name=member.organization_name or ""
            )
            for member in members
        ]
        
        # キャッシュ保存（10分）
        cache_data = [item.dict() for item in response_data]
        set_to_cache(cache_key, cache_data, ttl=600)
        
        return response_data
    
    except Exception as e:
        logger.error(f"部内メンバー取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === パフォーマンス監視・管理エンドポイント ===

@app.get("/admin/performance-metrics")
async def get_performance_metrics(limit: int = Query(100, le=1000)):
    """パフォーマンスメトリクス取得"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        metrics = redis_client.lrange("performance_metrics", 0, limit - 1)
        return [json.loads(metric) for metric in metrics]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/cache/clear")
async def clear_cache(pattern: str = Query("*")):
    """キャッシュクリア"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
        return {"message": f"Cleared {len(keys)} cache entries", "pattern": pattern}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/database-stats")
async def get_database_stats(db: Session = Depends(get_db)):
    """データベース統計情報"""
    try:
        stats_query = text("""
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables 
            WHERE tablename IN ('meetings', 'participants', 'users', 'organizations', 'agendas')
            ORDER BY tablename
        """)
        
        result = db.execute(stats_query)
        return [dict(row) for row in result.fetchall()]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === ヘルスチェック ===

@app.get("/health")
async def health_check():
    """詳細ヘルスチェック"""
    try:
        # データベース接続テスト
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Redis接続テスト
    redis_status = "healthy" if REDIS_AVAILABLE else "unavailable"
    if REDIS_AVAILABLE:
        try:
            redis_client.ping()
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "redis": redis_status,
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Meeting Management API - Optimized",
        "version": "2.0.0",
        "optimizations": [
            "N+1クエリ問題解決",
            "Redisキャッシュ機能",
            "ページネーション対応",
            "パフォーマンス監視",
            "自動インデックス作成",
            "コネクションプール最適化"
        ],
        "endpoints": {
            "optimized": [
                "/meeting_list_optimized",
                "/meeting_list_paginated"
            ],
            "legacy_compatible": [
                "/meeting_list",
                "/department_meetings",
                "/member_meetings"
            ],
            "admin": [
                "/admin/performance-metrics",
                "/admin/cache/clear",
                "/admin/database-stats"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )