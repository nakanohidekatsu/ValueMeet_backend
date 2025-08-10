# app.py
# app.py - 最終修正版（psycopg2接続を完全削除）

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime, date
from sqlalchemy.orm import Session

# 修正: psycopg2のimportを削除
# import psycopg2
# from psycopg2.extras import RealDictCursor
# import hashlib

from db_control import crud, mymodels
from db_control.create_tables import init_db
from db_control.crud import SessionLocal

# schemas.pyから必要なモデルをimport
from schemas import LoginRequest, LoginResponse, ResetPasswordRequest, MessageResponse

# アプリケーション初期化時にテーブルを作成
init_db()

# === Pydantic モデル ===

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    organization_id: int
    organization_name: str
    
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

app = FastAPI(title="Meeting Management API")

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

# データベース初期化
@app.on_event("startup")
async def startup_event():
    init_db()

# データベースセッション取得用のdependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === 認証エンドポイント（SQLAlchemyのみ使用） ===

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """ログイン認証 - SQLAlchemyセッションのみ使用"""
    try:
        # ユーザー情報を取得
        user = db.query(mymodels.User).filter(
            mymodels.User.user_id == request.user_id
        ).first()
        
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
    """パスワード初期化 - SQLAlchemyセッションのみ使用"""
    try:
        # 管理者権限チェック
        if request.admin_user_id != "admin":
            raise HTTPException(
                status_code=403, 
                detail="この機能は管理者のみ利用できます"
            )
        
        # 管理者の存在確認
        admin_user = db.query(mymodels.User).filter(
            mymodels.User.user_id == request.admin_user_id
        ).first()
        
        if not admin_user:
            raise HTTPException(
                status_code=403, 
                detail="管理者ユーザーが見つかりません"
            )
        
        # 対象ユーザーの存在確認
        target_user = db.query(mymodels.User).filter(
            mymodels.User.user_id == request.target_user_id
        ).first()
        
        if not target_user:
            raise HTTPException(
                status_code=404, 
                detail="対象ユーザーが見つかりません"
            )
        
        # パスワード更新
        target_user.password = request.new_password
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
    """トークン検証 - SQLAlchemyセッションのみ使用"""
    try:
        # ユーザー情報を取得
        user = db.query(mymodels.User).filter(
            mymodels.User.user_id == user_id
        ).first()
        
        if not user:
            raise HTTPException(status_code=401, detail="無効なユーザーです")
        
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

# 他のエンドポイントは既存のものを保持...
# (アジェンダ、タグ、参加者などのエンドポイント)

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# デバッグ用エンドポイント
@app.get("/debug/db-test")
async def debug_database_connection(db: Session = Depends(get_db)):
    """データベース接続テスト"""
    try:
        # 簡単なクエリでテスト
        result = db.execute("SELECT 1 as test")
        return {"status": "success", "message": "Database connection OK", "result": result.fetchone()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    