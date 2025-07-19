# app.py

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import json
import os

from db_control import crud, mymodels
from db_control.create_tables import init_db
from dotenv import load_dotenv
from typing import Optional

# アプリケーション初期化時にテーブルを作成
init_db()

# === Pydantic モデル ===
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


class Organization(BaseModel):
    organization_id: int
    organization_name: str

    class Config:
        orm_mode = True


class User(BaseModel):
    user_id: str
    name: str
    email: str
    organization_id: Optional[int] = None

    class Config:
        orm_mode = True


class Meeting(BaseModel):
    meeting_id: Optional[int] = None
    title: str
    meeting_type: Optional[str] = None
    meeting_mode: Optional[str] = None
    date_time: datetime
    created_by: Optional[str] = None

    class Config:
        orm_mode = True


class Agenda(BaseModel):
    agenda_id: Optional[int] = None
    meeting_id: Optional[int] = None
    purpose: Optional[str] = None
    topic: Optional[str] = None

    class Config:
        orm_mode = True


class Tag(BaseModel):
    tag_id: Optional[int] = None
    meeting_id: Optional[int] = None
    tag: str
    vector_embedding: List[float]

    class Config:
        orm_mode = True


class Participant(BaseModel):
    participant_id: Optional[int] = None
    meeting_id: Optional[int] = None
    user_id: Optional[str] = None
    role_type: Optional[str] = None

    class Config:
        orm_mode = True


class Todo(BaseModel):
    todo_id: Optional[int] = None
    meeting_id: Optional[int] = None
    assigned_to: Optional[str] = None
    content: str
    status: Optional[str] = None
    due_date: Optional[date] = None

    class Config:
        orm_mode = True


class Reminder(BaseModel):
    reminder_id: Optional[int] = None
    user_id: Optional[str] = None
    todo_id: Optional[int] = None
    remind_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class FileAttachment(BaseModel):
    file_id: str
    meeting_id: Optional[int] = None
    filename: Optional[str] = None
    file_url: Optional[str] = None

    class Config:
        orm_mode = True


class Transcript(BaseModel):
    transcript_id: Optional[int] = None
    meeting_id: Optional[int] = None
    speaker_id: Optional[str] = None
    content: str
    timestamp: Optional[datetime] = None

    class Config:
        orm_mode = True

    
app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .envファイルを読み込む
load_dotenv()
print("DATABASE_URL:", os.getenv("DATABASE_URL"))


@app.get("/")
def index():
    return {"message": "FastAPI top page!"}


# @app.get("/shouhin")
# def read_shouhin(CODE: int = Query(...)):
#     result = crud.myselect(mymodels.SHOUHIN, CODE)
#     if not result:
#         raise HTTPException(status_code=404, detail="商品が見つかりません")
#     items = json.loads(result)
#     return items[0] if items else None


# @app.post("/torihiki")
# def create_torihiki(data: TORIHIKI):
    # Pydantic モデル → dict
#     payload = data.dict(exclude_none=True, exclude={"TRD_ID"})
    # ORM モデルを正しく渡す
#     crud.myinsert_torihiki(mymodels.TORIHIKI, payload)
    # 最大 TRD_ID を取得
#     new_id = crud.myselect_TRD_ID(mymodels.TORIHIKI)
#     return {"TRD_ID": new_id}


# @app.post("/torimei")
# def create_torimei(data: TORIMEI):
        # by_alias=True で大文字キーに変換
#     values = data.model_dump(by_alias=True, exclude_unset=True)
#     crud.myinsert_torimei(mymodels.TORIMEI, values)
#     return {"status": "inserted"}


# 