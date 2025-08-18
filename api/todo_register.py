# api/todo_register.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import date

from db_control.connect import get_db
from db_control.mymodels import Todo

# 既存: todo_id / meeting_id / assigned_to / content / status / due_date

router = APIRouter(prefix="/api/todos", tags=["todos"])

class ExtractedTodoIn(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: Optional[str] = None        # "A000001" 等 or 名前
    due_date: Optional[str] = None        # "YYYY-MM-DD"
    priority: Optional[str] = None        # "高"/"中"/"低"

class RegisterReq(BaseModel):
    meeting_id: Optional[int] = None
    items: List[ExtractedTodoIn]

def _initial_status(p: Optional[str]) -> str:
    return "進行中" if p == "高" else "未着手"

def _normalize_assignee_to_user_id(v: Optional[str]) -> Optional[str]:
    """users.user_id 形式(A+数字)以外は None にしてFK違反を避ける。"""
    if not v:
        return None
    s = v.strip()
    if len(s) >= 2 and s[0].upper() == "A" and s[1:].isdigit():
        return s.upper()
    # TODO: 名前→user_id 解決をしたい場合はここで users を検索して返す
    return None

@router.post("/register")
def register_todos(req: RegisterReq, db: Session = Depends(get_db)):
    if not req.items:
        raise HTTPException(status_code=400, detail="items is empty")

    ids: list[int] = []
    try:
        for it in req.items:
            content = it.title if not it.description else f"{it.title}\n{it.description}"
            d = date.fromisoformat(it.due_date) if it.due_date else None

            todo = Todo(
                meeting_id=req.meeting_id,
                assigned_to=_normalize_assignee_to_user_id(it.assignee),
                content=content,
                status=_initial_status(it.priority),
                due_date=d,
            )
            db.add(todo)
            db.flush()
            ids.append(todo.todo_id)

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB登録に失敗: {e}")

    return {"inserted": len(ids), "todo_ids": ids}