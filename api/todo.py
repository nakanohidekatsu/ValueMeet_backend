from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from db_control.crud import SessionLocal

router = APIRouter(prefix="/todo", tags=["todo"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("")
def create_todo(meeting_id: int, title: str, db: Session = Depends(get_db)):
    db.execute(text("INSERT INTO todos (meeting_id, title) VALUES (:m, :t)"),
               {"m": meeting_id, "t": title})
    db.commit()
    return {"status": "ok"}

@router.get("")
def list_todos(db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT * FROM todos")).fetchall()
    return [dict(r) for r in rows]