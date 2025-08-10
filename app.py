# app.py - 超シンプル修正版（認証機能重点）

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime, date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from db_control import crud, mymodels
from db_control.create_tables import init_db
from db_control.crud import SessionLocal

# schemas.pyから必要なモデルをimport
from schemas import LoginRequest, LoginResponse, ResetPasswordRequest, MessageResponse

# アプリケーション初期化時にテーブルを作成
init_db()

# === 環境変数とデータベース設定 ===
load_dotenv()

# データベース設定（修正版）
def get_database_url():
    """環境変数からデータベースURLを取得（フォールバック付き）"""
    # 方法1: DATABASE_URLから取得
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # 方法2: 個別パラメータから構築
    host = os.getenv("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com")
    port = os.getenv("DB_PORT", "6543")
    database = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres.zdzdymwaessgxeojmtpb")
    password = os.getenv("DB_PASSWORD", "ValueMeet2025")
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

# SQLAlchemy設定
try:
    DATABASE_URL = get_database_url()
    print(f"使用するDATABASE_URL: {DATABASE_URL}")
    
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False  # SQLログを出力したい場合はTrue
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("✅ SQLAlchemy設定完了")
    
except Exception as e:
    print(f"❌ データベース設定エラー: {e}")
    raise e

# === Pydantic モデル ===

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    organization_id: int
    organization_name: str
    
class LoginRequest(BaseModel):
    user_id: str
    password: str

class LoginResponse(BaseModel):
    user_id: str
    name: str
    email: str
    organization_id: int
    organization_name: Optional[str] = None

class MessageResponse(BaseModel):
    message: str

class ResetPasswordRequest(BaseModel):
    admin_user_id: str
    target_user_id: str
    new_password: str

class DepartmentMember(BaseModel):
    user_id: str
    name: str
    organization_name: str

class MeetingListItem(BaseModel):
    meeting_id: int
    title: str
    meeting_type: Optional[str]
    meeting_mode: Optional[str]
    date_time: str
    name: str
    organization_name: str
    role_type: Optional[str]

class MeetingCreate(BaseModel):
    title: str
    meeting_type: Optional[str]
    meeting_mode: Optional[str]
    date_time: str
    created_by: str

class MeetingCreateResponse(BaseModel):
    meeting_id: int
    status: str = "success"

class AgendaCreate(BaseModel):
    meeting_id: int
    purpose: Optional[str] = None
    topic: Optional[str] = None
    purposes: Optional[List[str]] = None
    topics: Optional[List[str]] = None

class TagRegister(BaseModel):
    meeting_id: int
    tag: str

class TagsRegisterBatch(BaseModel):
    meeting_id: int
    tags: List[str]

class TagGenerateResponse(BaseModel):
    tags: List[str]

class RecommendUser(BaseModel):
    organization_name: str
    name: str
    user_id: str
    similarity_score: Optional[float] = None
    past_role: Optional[str] = None

class AttendCreate(BaseModel):
    meeting_id: int
    user_id: str
    role_type: Optional[str] = "participant"

class AttendCreateBatch(BaseModel):
    meeting_id: int
    participants: List[dict]

class NameSearchResult(BaseModel):
    organization_name: str
    name: str
    user_id: str
    email: Optional[str] = None
    
# === FastAPI アプリケーション ===
app = FastAPI(title="Meeting Management API - Simple Auth")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境変数の読み込み
load_dotenv()

# OpenAIクライアントの初期化
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# === データベースセッション ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === 認証エンドポイント ===

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """ログイン認証 - 純粋SQLクエリ使用"""
    try:
        # SQLクエリでユーザー情報を取得
        query = text("""
            SELECT u.user_id, u.name, u.email, u.organization_id, u.password, 
                   COALESCE(o.organization_name, '') as organization_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE u.user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": request.user_id})
        user = result.fetchone()
        
        if not user:
            raise HTTPException(
                status_code=401, 
                detail="ユーザーIDまたはパスワードが正しくありません"
            )
        
        # パスワード検証
        if user.password != request.password:
            raise HTTPException(
                status_code=401, 
                detail="ユーザーIDまたはパスワードが正しくありません"
            )
        
        return LoginResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            organization_id=user.organization_id or 0,
            organization_name=user.organization_name or ""
        )
    
        # パスワード検証
        if user.password != request.password:
            raise HTTPException(
                status_code=401, 
                detail="ユーザーIDまたはパスワードが正しくありません"
            )
        
        # 組織情報を取得
        organization = None
        if user.organization_id:
            organization = db.query(mymodels.Organization).filter(
                mymodels.Organization.organization_id == user.organization_id
            ).first()
        
        return LoginResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            organization_id=user.organization_id or 0,
            organization_name=organization.organization_name if organization else ""
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"ログイン処理中にエラーが発生しました: {str(e)}"
        )

@app.post("/auth/reset-password", response_model=MessageResponse)
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """パスワード初期化"""
    try:
        # 管理者権限チェック
        if request.admin_user_id != "admin":
            raise HTTPException(
                status_code=403, 
                detail="この機能は管理者のみ利用できます"
            )
        
        # 管理者存在確認
        admin_query = text("SELECT user_id FROM users WHERE user_id = :admin_id")
        admin_result = db.execute(admin_query, {"admin_id": request.admin_user_id})
        
        if not admin_result.fetchone():
            raise HTTPException(
                status_code=403, 
                detail="管理者ユーザーが見つかりません"
            )
        
        # 対象ユーザー存在確認
        user_query = text("SELECT user_id FROM users WHERE user_id = :user_id")
        user_result = db.execute(user_query, {"user_id": request.target_user_id})
        
        if not user_result.fetchone():
            raise HTTPException(
                status_code=404, 
                detail="対象ユーザーが見つかりません"
            )
        
        # パスワード更新
        update_query = text("""
            UPDATE users 
            SET password = :new_password 
            WHERE user_id = :user_id
        """)
        
        db.execute(update_query, {
            "new_password": request.new_password,
            "user_id": request.target_user_id
        })
        db.commit()
        
        return MessageResponse(
            message=f"ユーザー {request.target_user_id} のパスワードを初期化しました"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"パスワード初期化中にエラーが発生しました: {str(e)}"
        )

@app.get("/auth/validate-token", response_model=LoginResponse)
async def validate_token(user_id: str, db: Session = Depends(get_db)):
    """トークン検証"""
    try:
        query = text("""
            SELECT u.user_id, u.name, u.email, u.organization_id,
                   COALESCE(o.organization_name, '') as organization_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE u.user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": user_id})
        user = result.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="無効なユーザーです")
        
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
        raise HTTPException(
            status_code=500, 
            detail=f"トークン検証中にエラーが発生しました: {str(e)}"
        )

# === その他のエンドポイント（既存のものを保持） ===

@app.get("/usr_profile", response_model=UserProfileResponse)
async def get_usr_profile(user_id: str = Query(...)):
    """ユーザープロファイル取得API"""
    try:
        user = crud.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        organization = crud.get_organization_by_id(user.organization_id)
        organization_name = organization.organization_name if organization else ""
        
        return UserProfileResponse(
            user_id=user.user_id,
            name=user.name,
            organization_id=user.organization_id or 0,
            organization_name=organization_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meeting_list", response_model=List[MeetingListItem])
async def get_meeting_list(
    user_id: str = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    organization_id: Optional[int] = Query(None),
    meeting_type: Optional[str] = Query(None)
):
    """会議一覧取得API"""
    try:
        meeting_details = crud.get_meetings_by_user_with_details(
            user_id=user_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            organization_id=organization_id,
            meeting_type=meeting_type
        )
        
        meeting_list = []
        for meeting, creator_name, creator_organization_name, role_type in meeting_details:
            meeting_item = MeetingListItem(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                name=creator_name or "",
                organization_name=creator_organization_name or "",
                role_type=role_type
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meeting", response_model=MeetingCreateResponse)
async def create_meeting(meeting: MeetingCreate):
    """会議情報登録API"""
    try:
        if 'T' in meeting.date_time:
            date_time = datetime.fromisoformat(meeting.date_time.replace('Z', '+00:00'))
        else:
            date_time = datetime.strptime(meeting.date_time, "%Y/%m/%d %H:%M")
        
        meeting_id = crud.create_meeting(
            title=meeting.title,
            meeting_type=meeting.meeting_type,
            meeting_mode=meeting.meeting_mode,
            date_time=date_time,
            created_by=meeting.created_by
        )
        
        return MeetingCreateResponse(meeting_id=meeting_id, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === デバッグ用エンドポイント ===

@app.get("/debug/db-test")
async def debug_database_connection(db: Session = Depends(get_db)):
    """データベース接続テスト"""
    try:
        # 基本接続テスト
        result = db.execute(text("SELECT 1 as test, NOW() as current_time"))
        row = result.fetchone()
        
        # テーブル存在確認
        table_check = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'organizations')
        """))
        tables = [row[0] for row in table_check.fetchall()]
        
        return {
            "status": "success", 
            "message": "Database connection OK",
            "test_query": dict(row),
            "existing_tables": tables
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Database connection failed: {str(e)}"
        )

@app.get("/debug/user-check")
async def debug_user_check(user_id: str = "A000001", db: Session = Depends(get_db)):
    """特定ユーザーの存在確認"""
    try:
        query = text("""
            SELECT user_id, name, email, organization_id,
                   CASE WHEN password IS NOT NULL THEN 'SET' ELSE 'NULL' END as password_status
            FROM users 
            WHERE user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": user_id})
        user = result.fetchone()
        
        if user:
            return {
                "status": "found",
                "user": dict(user)
            }
        else:
            return {
                "status": "not_found",
                "user_id": user_id
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"User check failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "database_url_set": "DATABASE_URL" in os.environ
    }

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Meeting Management API", 
        "version": "1.0-simple",
        "endpoints": [
            "/auth/login",
            "/auth/reset-password", 
            "/auth/validate-token",
            "/debug/db-test",
            "/debug/user-check",
            "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    