# schemas.py
# schemas.py
from typing import Optional
from pydantic import BaseModel
from pydantic import ConfigDict

# 共通設定（ORM→Pydanticの変換を許可）
# Pydantic v2: from_attributes=True
class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# === 認証関連 DTO ===
class LoginRequest(BaseModel):
    """ログインリクエスト用のPydanticモデル（SQLAlchemyモデルではない）"""
    user_id: str
    password: str

class LoginResponse(BaseModel):
    """ログインレスポンス用のPydanticモデル"""
    user_id: str
    name: str
    email: str
    organization_id: int
    organization_name: Optional[str] = None

class ResetPasswordRequest(BaseModel):
    """パスワードリセットリクエスト用のPydanticモデル"""
    admin_user_id: str
    target_user_id: str
    new_password: str

class MessageResponse(BaseModel):
    """汎用メッセージレスポンス用のPydanticモデル"""
    message: str
    