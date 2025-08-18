# api/todo.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import date
import io, json, os, re  # ★変更: re を追加（コードフェンス除去で使用）

# DB
from db_control.connect import get_db
from db_control.mymodels import Todo  # ★変更: Todoモデルの正しい import を追加

# ▼ OpenAI & ファイル抽出
from openai import OpenAI
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

router = APIRouter(prefix="/api/todos", tags=["todos"])

# ---------- 抽出用スキーマ ----------
class TodoItem(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[str] = None        # "YYYY-MM-DD"
    priority: Optional[str] = None        # "高"/"中"/"低"
    source: Optional[dict] = None         # { filename, page, snippet }

class ExtractResponse(BaseModel):
    items: List[TodoItem]

# ---------- 登録用スキーマ ----------
class ExtractedTodoIn(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: Optional[str] = None        # "A000001" 等 or 名前
    due_date: Optional[str] = None        # "YYYY-MM-DD"
    priority: Optional[str] = None        # "高"/"中"/"低"

class RegisterReq(BaseModel):
    meeting_id: Optional[int] = None
    items: List[ExtractedTodoIn]

# ---------- OpenAI クライアント ----------
_client = None
def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

# ---------- ユーティリティ ----------
def read_any_text(file: UploadFile) -> str:
    """PDF/DOCX/TXT をいい感じに文字列化。未対応は 415 を返す。"""
    content = file.file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        return pdf_extract_text(io.BytesIO(content))
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=415, detail="Unsupported file type (pdf/docx/txt only)")

def initial_status(priority: Optional[str]) -> str:
    # 登録時の初期状態ルール：高→進行中 / それ以外→未着手
    return "進行中" if priority == "高" else "未着手"

def normalize_assignee_to_user_id(v: Optional[str]) -> Optional[str]:
    """users.user_id 形式(A+数字)以外は None にしてFK違反を避ける。必要ならここで名前→ID解決を実装。"""
    if not v:
        return None
    v2 = v.strip()
    if len(v2) >= 2 and v2[0].upper() == "A" and v2[1:].isdigit():
        return v2.upper()
    # TODO: usersテーブルから display_name などで検索→ user_id に変換するならここで
    return None  # ひとまずNoneにしてFKエラー回避

# ---------- 抽出 API ----------
@router.post("/extract", response_model=ExtractResponse)
async def extract_todos(
    file: UploadFile = File(...),
    language: str = Form("ja"),
    max_items: int = Form(50),
):
    print("★★★ extract_todos called! file:", file.filename)  # デバッグ用
    
    text = read_any_text(file)
    context = text[:6000]  # トークン節約

    # ★変更: JSONオブジェクト固定 ＆ 明確なスキーマ指定
    prompt = f"""
あなたは有能な秘書です。以下の会議資料テキストから「ToDo項目」を抽出し、
**次の厳密な JSON オブジェクト**のみを出力してください。前後の説明やコードフェンスは禁止。

出力スキーマ:
{{
  "items": [
    {{
      "title": "string",
      "description": "string|null",
      "assignee": "string|null",
      "due_date": "YYYY-MM-DD|null",
      "priority": "高|中|低",
      "source": {{
        "filename": "string",
        "page": null,
        "snippet": "string|null"
      }}
    }}
  ]
}}

制約:
- 最大 {max_items} 件
- 欠損は null を入れる
- priority は必ず "高"|"中"|"低"
- language は {language}
- source.filename は "{file.filename}"

会議資料:
{context}
""".strip()

    try:
        resp = openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},  # ★変更: JSON固定
        )
        content = resp.choices[0].message.content or ""

        # ★変更: 念のためコードフェンス除去
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)

        data = json.loads(cleaned)

        # ★変更: 配列単体が返るパターンの救済
        if isinstance(data, list):
            data = {"items": data}

        items = data.get("items", [])
        if not isinstance(items, list):
            raise ValueError("`items` が配列ではありません")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLMレスポンスがJSONとして解析できません: {e}")

    # filename を補完
    for it in items:
        it.setdefault("source", {})
        it["source"].setdefault("filename", file.filename)

    return {"items": items}

# ---------- 登録 API ----------
@router.post("/register")
def register_todos(req: RegisterReq, db: Session = Depends(get_db)):
    if not req.items:
        raise HTTPException(status_code=400, detail="items is empty")

    ids: List[int] = []
    try:
        for it in req.items:
            # content は title + 改行 + description（descriptionがあれば）
            content = it.title if not it.description else f"{it.title}\n{it.description}"
            d = date.fromisoformat(it.due_date) if it.due_date else None

            todo = Todo(
                meeting_id=req.meeting_id,
                assigned_to=normalize_assignee_to_user_id(it.assignee),  # FK安全
                content=content,
                status=initial_status(it.priority),
                due_date=d,
            )
            db.add(todo)
            db.flush()              # INSERTせずにPK採番を確定させる（トランザクション内）
            ids.append(todo.todo_id)

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB登録に失敗: {e}")

    return {"inserted": len(ids), "todo_ids": ids}

