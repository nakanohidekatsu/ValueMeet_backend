# app.py

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import os
import json
import openai

from db_control import crud, mymodels
from db_control.create_tables import init_db
from dotenv import load_dotenv
from typing import Optional, List

from mymodels import User, Organization  # SQLAlchemy models
from database import SessionLocal
import crud

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

class AgendaCreate(BaseModel):
    meeting_id: int
    purpose: Optional[str]
    topic: Optional[str]

class TagRegister(BaseModel):
    meeting_id: int
    tag: str
    vector_embedding: List[float]

class TagGenerateResponse(BaseModel):
    tags: List[str]

class RecommendUser(BaseModel):
    organization_name: str
    name: str
    user_id: str

class AttendCreate(BaseModel):
    meeting_id: int
    user_id: str
    role_type: Optional[str]

class NameSearchResult(BaseModel):
    organization_name: str
    name: str
    user_id: str

# === FastAPI アプリケーション ===

app = FastAPI(title="Meeting Management API")


# レスポンスモデル
class TagGenerateResponse(BaseModel):
    tags: List[str]

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

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# データベース初期化
@app.on_event("startup")
async def startup_event():
    init_db()

# === API エンドポイント ===

@app.get("/usr_profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str = Query(...)):
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
                date_time=meeting.date_time.strftime("%Y/%m/%d %H:%M"),
                name=creator.name if creator else "",
                organization_name=creator_organization.organization_name if creator_organization else "",
                role_type=role_type
            )
            meeting_list.append(meeting_item)
        
        return meeting_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meeting")
async def create_meeting(meeting: MeetingCreate):
    """
    会議情報登録API
    会議情報を登録する
    """
    try:
        # 日時文字列をdatetimeオブジェクトに変換
        date_time = datetime.strptime(meeting.date_time, "%Y/%m/%d %H:%M")
        
        meeting_id = crud.create_meeting(
            title=meeting.title,
            meeting_type=meeting.meeting_type,
            meeting_mode=meeting.meeting_mode,
            date_time=date_time,
            created_by=meeting.created_by
        )
        
        return {"meeting_id": meeting_id, "message": "Meeting created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agenda")
async def create_agenda(agenda: AgendaCreate):
    """
    アジェンダ登録API
    会議のレジュメを登録する
    """
    try:
        agenda_id = crud.create_agenda(
            meeting_id=agenda.meeting_id,
            purpose=agenda.purpose,
            topic=agenda.topic
        )
        
        return {"agenda_id": agenda_id, "message": "Agenda created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tag_register")
async def register_tag(tag_data: TagRegister):
    """
    タグ登録API
    ベクトル化したタグを登録
    """
    try:
        tag_id = crud.create_tag(
            meeting_id=tag_data.meeting_id,
            tag=tag_data.tag,
            vector_embedding=tag_data.vector_embedding
        )
        
        return {"tag_id": tag_id, "message": "Tag registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tag_generate", response_model=TagGenerateResponse)
async def generate_tags(topic: str = Query(..., description="抽出対象の文章")):
    """
    タグ生成API
    レジュメからChatGPTを使ってキーワードを抽出する
    """
    # プロンプト作成
    prompt = (
        "以下の文章から主要なキーワードを5つ、"
        "必ず**純粋な JSON 配列**（例: [\"A\",\"B\",\"C\",\"D\",\"E\"]）で"
        "それ以外のコメントや説明を付けずに出力してください。\n"
        f"文章:\n{topic}"
    )
    try:
        # ChatGPT API呼び出し
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts keywords."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        # 応答テキスト取得
        content = response.choices[0].message.content.strip()
        # JSONパース
        tags = json.loads(content)
        if not isinstance(tags, list):
            raise ValueError("キーワードがリスト形式ではありません: " + content)

        return TagGenerateResponse(tags=tags)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ生成に失敗しました: {str(e)}")

@app.get("/recommend", response_model=List[RecommendUser])
async def get_recommendations(
    tag: str = Query(..., description="基準となるキーワード"),
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
    tag: キーワード
    top_k: 上位何件の類似会議IDを参照するか
    """
    # （以降は前回ご案内の「ベクトル化→ベクトル検索→参加者取得」処理）
    # 1) ChatGPT API でタグをベクトル化
    try:
        embed_resp = openai.Embedding.create(
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
        similar_meeting_ids = crud.find_meeting_ids_by_tag_vector(db, query_vector, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"類似タグ検索に失敗: {e}")

    # 3) 参加者取得
    try:
        users = crud.get_users_by_meeting_ids(db, similar_meeting_ids)
        result: List[RecommendUser] = []
        for user in users:
            org = crud.get_organization_by_id(db, user.organization_id)
            result.append(
                RecommendUser(
                    organization_name=org.organization_name if org else "",
                    name=user.name,
                    user_id=user.user_id
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
            role_type=attend.role_type
        )
        
        return {"participant_id": participant_id, "message": "Participant registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/name_search", response_model=List[NameSearchResult])
async def search_by_name(name: str = Query(...)):
    """
    名前検索API
    参加者指定の際にユーザー情報から名前検索する
    """
    try:
        users = crud.search_users_by_name(name)
        
        result = []
        for user in users:
            organization = crud.get_organization_by_id(user.organization_id)
            search_result = NameSearchResult(
                organization_name=organization.organization_name if organization else "",
                name=user.name,
                user_id=user.user_id
            )
            result.append(search_result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === ヘルスチェック ===

@app.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    