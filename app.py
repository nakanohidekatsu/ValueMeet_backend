# app.py

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
from openai import OpenAI

# from . import crud, mymodels
from db_control import crud, mymodels
from db_control.create_tables import init_db
from dotenv import load_dotenv
from typing import Optional, List

from db_control.crud import SessionLocal
from db_control import crud
from sqlalchemy.orm import Session

# アプリケーション初期化時にテーブルを作成
init_db()

# === Pydantic モデル ===
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


# === Pydantic モデル ===

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    organization_id: int
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

# 修正: アジェンダに配列対応
class AgendaCreate(BaseModel):
    meeting_id: int
    purpose: Optional[str] = None  # "||"で区切られた複数の目的
    topic: Optional[str] = None    # "||"で区切られた複数のトピック
    purposes: Optional[List[str]] = None  # 目的の配列（新規追加）
    topics: Optional[List[str]] = None    # トピックの配列（新規追加）

class TagRegister(BaseModel):
    meeting_id: int
    tag: str

# 新規追加: 複数タグ一括登録
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

# 新規追加: 複数参加者一括登録
class AttendCreateBatch(BaseModel):
    meeting_id: int
    participants: List[dict]  # [{"user_id": "xxx", "role_type": "participant"}]

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

# === API エンドポイント ===

@app.get("/user_profile", response_model=UserProfileResponse)
async def get_usr_profile(user_id: str = Query(...)):
    """
    ユーザープロファイル取得API
    起動時（ログイン時）に初期画面に必要な情報を取得
    """
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
    """
    会議一覧取得API
    ログインユーザーが参加する会議の一覧を表示
    """
    try:
        meetings = crud.get_meetings_by_user(
            user_id=user_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            organization_id=organization_id,
            meeting_type=meeting_type
        )
        
        meeting_list = []
        for meeting in meetings:
            # 主催者情報取得
            creator = crud.get_user_by_id(meeting.created_by)
            creator_organization = crud.get_organization_by_id(creator.organization_id) if creator else None
            
            # 参加者の役割取得
            participant = crud.get_participant_role(meeting.meeting_id, user_id)
            role_type = participant.role_type if participant else None
            
            meeting_item = MeetingListItem(
                meeting_id=meeting.meeting_id,
                title=meeting.title,
                meeting_type=meeting.meeting_type,
                meeting_mode=meeting.meeting_mode,
                date_time=meeting.date_time.strftime("%Y-%m-%dT%H:%M:00"),
                name=creator.name if creator else "",
                organization_name=creator_organization.organization_name if creator_organization else "",
                role_type=role_type
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meeting", response_model=MeetingCreateResponse)
async def create_meeting(meeting: MeetingCreate):
    """
    会議情報登録API
    会議情報を登録する
    """
    try:
        # 日時文字列をdatetimeオブジェクトに変換
        # ISO形式に対応
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

@app.post("/agenda")
async def create_agenda(agenda: AgendaCreate):
    """
    アジェンダ登録API
    会議のレジュメを登録する（配列対応版）
    """
    try:
        # 配列が渡された場合は"||"で結合
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
    タグ一括登録API（新規追加）
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
    参加者一括登録API（新規追加）
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

# === デバッグ用エンドポイント（開発環境のみ） ===

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

# === ヘルスチェック ===

@app.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)