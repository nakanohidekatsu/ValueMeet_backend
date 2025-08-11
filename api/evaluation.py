from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from db_control.crud import SessionLocal

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("")
def create_eval(meeting_id: int, user_id: str, score: int, db: Session = Depends(get_db)):
    db.execute(text(
        "INSERT INTO meeting_evaluations (meeting_id, user_id, score) VALUES (:m, :u, :s)"
    ), {"m": meeting_id, "u": user_id, "s": score})
    db.commit()
    return {"status": "ok"}

@router.get("")
def list_evals(db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT * FROM meeting_evaluations")).fetchall()
    return [dict(r) for r in rows]