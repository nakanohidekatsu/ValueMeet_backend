# api/evaluation.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel, conint
from typing import Optional, List, Dict, Any

# あなたの既存セッションファクトリ
from db_control.crud import SessionLocal

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# --- DB Session dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Request model ---
class EvaluationInput(BaseModel):
    meeting_id: int
    user_id: str                              # 本番ではJWTから取得予定
    satisfaction: conint(ge=1, le=5)
    rejoin_intent: conint(ge=1, le=5)
    self_contribution: conint(ge=1, le=5)
    facilitator_rating: conint(ge=1, le=5)
    comment: Optional[str] = None

# --- Create/Update (UPSERT) ---
@router.post("")
def create_or_update_eval(e: EvaluationInput, db: Session = Depends(get_db)) -> Dict[str, Any]:
    # users.user_id の存在チェック
    user_exists = db.execute(
        text("SELECT 1 FROM users WHERE user_id = :u"),
        {"u": e.user_id}
    ).scalar()
    if not user_exists:
        raise HTTPException(status_code=400, detail=f"user_id '{e.user_id}' not found")

    # meetings.meeting_id の存在チェック
    meeting_exists = db.execute(
        text("SELECT 1 FROM meetings WHERE meeting_id = :m"),
        {"m": e.meeting_id}
    ).scalar()
    if not meeting_exists:
        raise HTTPException(status_code=400, detail=f"meeting_id '{e.meeting_id}' not found")

    # タイムスタンプ列を使わないUPSERT
    db.execute(text("""
        INSERT INTO meeting_evaluations (
          meeting_id, user_id, satisfaction, rejoin_intent,
          self_contribution, facilitator_rating, comment
        ) VALUES (
          :meeting_id, :user_id, :satisfaction, :rejoin_intent,
          :self_contribution, :facilitator_rating, :comment
        )
        ON CONFLICT (meeting_id, user_id) DO UPDATE SET
          satisfaction       = EXCLUDED.satisfaction,
          rejoin_intent      = EXCLUDED.rejoin_intent,
          self_contribution  = EXCLUDED.self_contribution,
          facilitator_rating = EXCLUDED.facilitator_rating,
          comment            = EXCLUDED.comment
    """), e.model_dump())

    db.commit()
    return {"status": "ok", "mode": "upsert"}

# --- Read (list all) ---
@router.get("")
def list_evals(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    rows = db.execute(text("SELECT * FROM meeting_evaluations")).mappings().all()
    return [dict(r) for r in rows]