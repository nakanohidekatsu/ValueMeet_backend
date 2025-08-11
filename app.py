# app.py

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime, date, time
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
    purpose: Optional[str] = None
    status: Optional[str] = "draft"
    
class MeetingCreate(BaseModel):
    title: str
    description: Optional[str] = None  # 会議概要を追加
    meeting_type: Optional[str] = None
    meeting_mode: Optional[str] = None
    priority: Optional[str] = None  # 優先度を追加
    date_time: str
    end_time: Optional[str] = None  # 終了時間を追加
    created_by: str
    status: Optional[str] = "draft"  # ステータスを追加

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
    
class MeetingDetailResponse(BaseModel):
    meeting_id: int
    title: str
    description: Optional[str] = None
    meeting_type: Optional[str] = None
    meeting_mode: Optional[str] = None
    priority: Optional[str] = None
    date_time: str
    end_time: Optional[str] = None
    created_by: str
    status: Optional[str] = None

class AgendaResponse(BaseModel):
    agenda_id: int
    meeting_id: int
    purpose: Optional[str] = None
    topic: Optional[str] = None

class ParticipantResponse(BaseModel):
    user_id: str
    name: str
    organization_name: str
    role_type: str
    email: Optional[str] = None

# === FastAPI アプリケーション ===
app = FastAPI(title="Meeting Management API - Simple Auth")

# きょん：APIルーター用コード
from api.todo import router as todo_router
from api.evaluation import router as evaluation_router

app.include_router(todo_router)
app.include_router(evaluation_router)

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
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# キー設定エラーバイパスのため
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠ OPENAI_API_KEY が設定されていません。OpenAI連携機能は無効になります。")

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

# === 会議管理エンドポイント ===

@app.post("/meeting", response_model=MeetingCreateResponse)
async def create_meeting(meeting: MeetingCreate, db: Session = Depends(get_db)):
    """会議情報登録API（拡張版）"""
    try:
        # 日時の処理
        if 'T' in meeting.date_time:
            date_time = datetime.fromisoformat(meeting.date_time.replace('Z', '+00:00'))
        else:
            date_time = datetime.strptime(meeting.date_time, "%Y/%m/%d %H:%M")
        
        # 終了時間の処理
        end_time = None
        if meeting.end_time:
            # HH:MM形式の文字列をtime型に変換
            try:
                time_parts = meeting.end_time.split(':')
                end_time = time(int(time_parts[0]), int(time_parts[1]))
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="終了時間の形式が正しくありません (HH:MM)")

        # SQLを直接使用して会議を作成（新しいフィールドを含む）
        insert_query = text("""
            INSERT INTO meetings (
                title, description, meeting_type, meeting_mode, priority,
                date_time, end_time, created_by, status, created_at
            ) VALUES (
                :title, :description, :meeting_type, :meeting_mode, :priority,
                :date_time, :end_time, :created_by, :status, :created_at
            ) RETURNING meeting_id
        """)
        
        result = db.execute(insert_query, {
            "title": meeting.title,
            "description": meeting.description,
            "meeting_type": meeting.meeting_type,
            "meeting_mode": meeting.meeting_mode,
            "priority": meeting.priority,
            "date_time": date_time,
            "end_time": end_time,
            "created_by": meeting.created_by,
            "status": meeting.status or "draft",
            "created_at": datetime.now()
        })
        
        meeting_id = result.fetchone()[0]
        db.commit()
        
        return MeetingCreateResponse(meeting_id=meeting_id, status="success")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"会議作成に失敗しました: {str(e)}")

@app.get("/meeting/{meeting_id}", response_model=MeetingDetailResponse)
async def get_meeting_detail(meeting_id: int, db: Session = Depends(get_db)):
    """
    会議詳細取得API
    指定されたIDの会議の詳細情報を取得
    """
    try:
        # 会議情報を取得
        query = text("""
            SELECT meeting_id, title, description, meeting_type, meeting_mode, 
                   priority, date_time, end_time, created_by, status
            FROM meetings 
            WHERE meeting_id = :meeting_id
        """)
        
        result = db.execute(query, {"meeting_id": meeting_id})
        meeting = result.fetchone()
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        return MeetingDetailResponse(
            meeting_id=meeting.meeting_id,
            title=meeting.title,
            description=meeting.description,
            meeting_type=meeting.meeting_type,
            meeting_mode=meeting.meeting_mode,
            priority=meeting.priority,
            date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
            end_time=meeting.end_time.strftime("%H:%M") if meeting.end_time else None,
            created_by=meeting.created_by,
            status=meeting.status
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"会議詳細取得中にエラーが発生しました: {str(e)}"
        )

@app.get("/meeting/{meeting_id}/participants", response_model=List[ParticipantResponse])
async def get_meeting_participants(meeting_id: int, db: Session = Depends(get_db)):
    """
    参加者取得API
    指定された会議の参加者一覧を取得
    """
    try:
        query = text("""
            SELECT p.user_id, u.name, o.organization_name, p.role_type, u.email
            FROM participants p
            LEFT JOIN users u ON p.user_id = u.user_id
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE p.meeting_id = :meeting_id
            ORDER BY u.name
        """)
        
        result = db.execute(query, {"meeting_id": meeting_id})
        participants = result.fetchall()
        
        return [
            ParticipantResponse(
                user_id=p.user_id,
                name=p.name,
                organization_name=p.organization_name or "",
                role_type=p.role_type,
                email=p.email
            )
            for p in participants
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"参加者取得中にエラーが発生しました: {str(e)}"
        )
        
@app.get("/meeting/{meeting_id}/agenda", response_model=AgendaResponse)
async def get_meeting_agenda(meeting_id: int, db: Session = Depends(get_db)):
    """
    アジェンダ取得API
    指定された会議のアジェンダを取得
    """
    try:
        query = text("""
            SELECT agenda_id, meeting_id, purpose, topic
            FROM agendas 
            WHERE meeting_id = :meeting_id
        """)
        
        result = db.execute(query, {"meeting_id": meeting_id})
        agenda = result.fetchone()
        
        if not agenda:
            raise HTTPException(status_code=404, detail="Agenda not found")
        
        return AgendaResponse(
            agenda_id=agenda.agenda_id,
            meeting_id=agenda.meeting_id,
            purpose=agenda.purpose,
            topic=agenda.topic
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"アジェンダ取得中にエラーが発生しました: {str(e)}"
        )

# === その他既存エンドポイント ===

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
            agenda = crud.get_agenda_by_meeting_id(meeting.meeting_id)
            purpose = agenda.purpose if agenda else None
            meeting_item = MeetingListItem(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                name=creator_name or "",
                organization_name=creator_organization_name or "",
                role_type=role_type,
                purpose=purpose,
                status=meeting.status if hasattr(meeting, 'status') and meeting.status else "scheduled"
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/department_members", response_model=List[DepartmentMember])
async def get_department_members(
    organization_id: int = Query(...),
    db: Session = Depends(get_db)
):
    """部内メンバー一覧取得API"""
    try:
        # 組織に所属するユーザーを取得
        query = text("""
            SELECT u.user_id, u.name, o.organization_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            WHERE u.organization_id = :organization_id
            ORDER BY u.name
        """)
        
        result = db.execute(query, {"organization_id": organization_id})
        members = result.fetchall()
        
        return [
            DepartmentMember(
                user_id=member.user_id,
                name=member.name,
                organization_name=member.organization_name or ""
            )
            for member in members
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"部内メンバー取得中にエラーが発生しました: {str(e)}"
        )

@app.get("/department_meetings", response_model=List[MeetingListItem])  
async def get_department_meetings(
    organization_id: int = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """部内会議一覧取得API（参加者数計算修正版）"""
    try:
        # 組織に所属するユーザーが作成または参加した会議を取得
        query = text("""
            SELECT DISTINCT m.meeting_id, m.title, m.meeting_type, m.meeting_mode, 
                   m.date_time, m.status, u.name, o.organization_name, p.role_type
            FROM meetings m
            LEFT JOIN users u ON m.created_by = u.user_id
            LEFT JOIN organizations o ON u.organization_id = o.organization_id
            LEFT JOIN participants p ON m.meeting_id = p.meeting_id
            LEFT JOIN users pu ON p.user_id = pu.user_id
            WHERE (u.organization_id = :organization_id 
                   OR pu.organization_id = :organization_id)
            ORDER BY m.date_time
        """)
        
        result = db.execute(query, {"organization_id": organization_id})
        meetings = result.fetchall()
        
        meeting_list = []
        for meeting in meetings:
            # 実際の参加者数を計算
            participant_count = crud.get_participant_count(db, meeting.meeting_id)
            
            # アジェンダから目的を取得
            agenda = crud.get_agenda_by_meeting_id(meeting.meeting_id)
            purpose = agenda.purpose if agenda else None
            
            meeting_item = MeetingListItem(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                name=meeting.name or "",
                organization_name=meeting.organization_name or "",
                role_type=meeting.role_type,
                purpose=purpose,
                status=meeting.status if hasattr(meeting, 'status') and meeting.status else "scheduled",
                # 修正: 実際の参加者数を設定
                participants=participant_count
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"部内会議一覧取得中にエラーが発生しました: {str(e)}"
        )


@app.get("/member_meetings", response_model=List[MeetingListItem])
async def get_member_meetings(
    member_id: str = Query(...),
    start_datetime: Optional[str] = Query(None),
    end_datetime: Optional[str] = Query(None),
    meeting_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """担当者の会議一覧取得API（参加者数計算修正版）"""
    try:
        meeting_details = crud.get_meetings_by_user_with_details(
            user_id=member_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            meeting_type=meeting_type
        )
        
        meeting_list = []
        for meeting, creator_name, creator_organization_name, role_type in meeting_details:
            # 実際の参加者数を計算
            participant_count = crud.get_participant_count(db, meeting.meeting_id)
            
            # アジェンダから目的を取得
            agenda = crud.get_agenda_by_meeting_id(meeting.meeting_id)
            purpose = agenda.purpose if agenda else None
            
            meeting_item = MeetingListItem(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                name=creator_name or "",
                organization_name=creator_organization_name or "",
                role_type=role_type,
                purpose=purpose,
                status=meeting.status if hasattr(meeting, 'status') and meeting.status else "scheduled",
                # 修正: 実際の参加者数を設定
                participants=participant_count
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"担当者会議一覧取得中にエラーが発生しました: {str(e)}"
        )

# 1. アジェンダ登録APIの修正（配列対応）
@app.post("/agenda")
async def create_agenda(agenda: AgendaCreate):
    """
    アジェンダ登録API
    会議のレジュメを登録する（配列対応版）
    """
    try:
        # 配列が渡された場合は"||"で結合して保存
        if agenda.purposes and isinstance(agenda.purposes, list):
            purpose_str = "||".join(filter(None, agenda.purposes))
        else:
            purpose_str = agenda.purpose
        
        if agenda.topics and isinstance(agenda.topics, list):
            topic_str = "||".join(filter(None, agenda.topics))
        else:
            topic_str = agenda.topic
        
        agenda_id = crud.create_agenda(
            meeting_id=agenda.meeting_id,
            purpose=purpose_str,
            topic=topic_str
        )
        
        return {"agenda_id": agenda_id, "status": "success", "message": "Agenda created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tag_register")
async def register_tag(tag_data: TagRegister):
    """
    タグ登録API
    タグをベクトル化して登録
    """
    try:
        # タグをベクトル化
        embed_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[tag_data.tag]
        )
        vector = embed_resp.data[0].embedding

        # CRUD で保存
        tag_id = crud.create_tag(
            meeting_id=tag_data.meeting_id,
            tag=tag_data.tag,
            vector_embedding=vector
        )
        return {"tag_id": tag_id, "status": "success", "message": "Tag registered successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ登録に失敗しました: {e}")

@app.post("/tags_register_batch")
async def register_tags_batch(tag_data: TagsRegisterBatch):
    """
    タグ一括登録API
    複数のタグをまとめてベクトル化して登録
    """
    try:
        registered_tags = []
        
        for tag in tag_data.tags:
            # 各タグをベクトル化
            embed_resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=[tag]
            )
            vector = embed_resp.data[0].embedding
            
            # CRUD で保存
            tag_id = crud.create_tag(
                meeting_id=tag_data.meeting_id,
                tag=tag,
                vector_embedding=vector
            )
            registered_tags.append({"tag_id": tag_id, "tag": tag})
        
        return {
            "status": "success",
            "message": f"{len(registered_tags)} tags registered successfully",
            "tags": registered_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ一括登録に失敗しました: {e}")

@app.get("/tag_generate", response_model=TagGenerateResponse)
async def generate_tags(topic: str = Query(..., description="抽出対象の文章")):
    """
    タグ生成API
    レジュメからChatGPTを使ってキーワードを抽出する
    """
    # プロンプト作成
    prompt = (
        "以下の文章から重要なキーワードを5〜8個抽出してください。"
        "会議の内容を的確に表すキーワードを選んでください。"
        "必ず**純粋な JSON 配列**（例: [\"A\",\"B\",\"C\",\"D\",\"E\"]）で"
        "それ以外のコメントや説明を付けずに出力してください。\n"
        f"文章:\n{topic}"
    )
    try:
        # ChatGPT API呼び出し
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts keywords from meeting topics in Japanese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        # 応答テキスト取得
        content = response.choices[0].message.content.strip()
        # JSONパース
        tags = json.loads(content)
        if not isinstance(tags, list):
            raise ValueError("キーワードがリスト形式ではありません: " + content)

        # 最大8個まで制限
        tags = tags[:8]

        return TagGenerateResponse(tags=tags)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ生成に失敗しました: {str(e)}")

from sqlalchemy import select, func

@app.get("/recommend", response_model=List[RecommendUser])
async def get_recommendations(
    tag: str = Query(..., description="基準となるキーワード（スペース区切りで複数可）"),
    top_k: int = Query(
        5,
        title="結果件数",
        description="類似会議として返す上位件数。大きいほど広く拾います",
        ge=1,
        le=100
    )
):
    """
    おすすめ参加者API
    tag: キーワード（スペース区切りで複数指定可能）
    top_k: 上位何件の類似会議IDを参照するか
    """
    # 1) タグをベクトル化
    try:
        embed_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[tag]
        )
        query_vector = embed_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ埋め込みの取得に失敗: {e}")

    # 2) DB からベクトル検索で近傍タグを取得
    try:
        db = SessionLocal()
        # crud 側で pgvector の <-> 演算子を使った検索を実装
        similar_meeting_ids = crud.find_meeting_ids_by_tag_vector(db, query_vector, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"類似タグ検索に失敗: {e}")

    # 3) 類似会議の参加者を取得
    try:
        users = crud.get_users_by_meeting_ids(db, similar_meeting_ids)
        
        # 重複を除去し、ユーザー情報を整形
        seen_users = set()
        result: List[RecommendUser] = []
        
        for user in users:
            if user.user_id not in seen_users:
                seen_users.add(user.user_id)
                org = crud.get_organization_by_id(user.organization_id)
                
                # 過去の役割を取得（オプション）
                past_role = None
                for meeting_id in similar_meeting_ids:
                    participant = crud.get_participant_role(meeting_id, user.user_id)
                    if participant:
                        past_role = participant.role_type
                        break
                
                result.append(
                    RecommendUser(
                        organization_name=org.organization_name if org else "",
                        name=user.name,
                        user_id=user.user_id,
                        past_role=past_role
                    )
                )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"参加者取得に失敗: {e}")
    finally:
        db.close()

@app.post("/attend")
async def create_attendance(attend: AttendCreate):
    """
    参加者登録API
    会議の参加者を登録する
    """
    try:
        participant_id = crud.create_participant(
            meeting_id=attend.meeting_id,
            user_id=attend.user_id,
            role_type=attend.role_type or "participant"
        )
        
        return {
            "participant_id": participant_id,
            "status": "success",
            "message": "Participant registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attend_batch")
async def create_attendance_batch(attend_batch: AttendCreateBatch):
    """
    参加者一括登録API
    複数の参加者をまとめて登録する
    """
    try:
        registered_participants = []
        
        for participant in attend_batch.participants:
            participant_id = crud.create_participant(
                meeting_id=attend_batch.meeting_id,
                user_id=participant.get("user_id"),
                role_type=participant.get("role_type", "participant")
            )
            registered_participants.append({
                "participant_id": participant_id,
                "user_id": participant.get("user_id"),
                "role_type": participant.get("role_type", "participant")
            })
        
        return {
            "status": "success",
            "message": f"{len(registered_participants)} participants registered successfully",
            "participants": registered_participants
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/name_search", response_model=List[NameSearchResult])
async def search_by_name(name: str = Query(..., description="検索する名前（部分一致）")):
    """
    名前検索API
    参加者指定の際にユーザー情報から名前検索する（部分一致検索）
    """
    try:
        # 部分一致で検索
        users = crud.search_users_by_name(name)
        
        result = []
        for user in users:
            organization = crud.get_organization_by_id(user.organization_id)
            
            # ユーザーのメールアドレスも取得（あれば）
            email = getattr(user, 'email', None)
            
            search_result = NameSearchResult(
                organization_name=organization.organization_name if organization else "",
                name=user.name,
                user_id=user.user_id,
                email=email
            )
            result.append(search_result)
        
        # 名前でソート
        result.sort(key=lambda x: x.name)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/meeting/{meeting_id}")
async def delete_meeting(meeting_id: int):
    """
    会議削除API
    指定された会議を削除する
    """
    try:
        # 会議の存在確認
        meeting = crud.get_meeting_by_id(meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # 関連データの削除
        crud.delete_meeting_and_related_data(meeting_id)
        
        return {"status": "success", "message": "Meeting deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === デバッグ用エンドポイント ===

@app.get("/debug/meetings")
async def debug_get_all_meetings():
    """
    デバッグ用: 全会議情報を取得
    """
    try:
        db = SessionLocal()
        meetings = crud.get_all_meetings(db)
        return {"total": len(meetings), "meetings": meetings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/debug/tags/{meeting_id}")
async def debug_get_meeting_tags(meeting_id: int):
    """
    デバッグ用: 特定会議のタグを取得
    """
    try:
        db = SessionLocal()
        tags = crud.get_tags_by_meeting_id(db, meeting_id)
        return {"meeting_id": meeting_id, "tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

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
            AND table_name IN ('users', 'organizations', 'meetings')
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

@app.get("/debug/meeting/{meeting_id}/participant-count")
async def debug_get_participant_count(meeting_id: int, db: Session = Depends(get_db)):
    """デバッグ用: 特定会議の参加者数と詳細を確認"""
    try:
        # 参加者数を取得
        participant_count = crud.get_participant_count(db, meeting_id)
        
        # 参加者の詳細を取得
        participant_details = crud.get_participant_details_by_meeting_id(meeting_id, db)
        
        return {
            "meeting_id": meeting_id,
            "participant_count": participant_count,
            "participants": [
                {
                    "user_id": p.user_id,
                    "name": p.name,
                    "organization_name": p.organization_name or "",
                    "role_type": p.role_type
                }
                for p in participant_details
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"参加者情報取得エラー: {str(e)}"
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
        "version": "1.1-enhanced",
        "new_features": [
            "会議概要 (description)",
            "優先度 (priority)", 
            "終了時間 (end_time)",
            "ステータス (status)"
        ],
        "endpoints": [
            "/auth/login",
            "/auth/reset-password", 
            "/auth/validate-token",
            "/meeting (POST) - Enhanced",
            "/meeting/{meeting_id} (GET)",
            "/meeting/{meeting_id}/agenda (GET)",
            "/meeting/{meeting_id}/participants (GET)",
            "/debug/db-test",
            "/debug/user-check",
            "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)