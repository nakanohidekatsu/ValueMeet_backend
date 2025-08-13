# db_control/create_tables.py

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from .mymodels import Base

# 環境変数の読み込み
load_dotenv()

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
DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """データベースの初期化とテーブル作成"""
    try:
        print("📊 データベーステーブルを初期化中...")
        
        # pgvector拡張を有効にする
        with engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("✅ pgvector拡張を有効化しました")
            except Exception as e:
                print(f"⚠️  pgvector拡張の有効化に失敗: {e}")
                # pgvectorが既に有効化されている場合は続行
        
        # 新しいフィールドを既存テーブルに追加（マイグレーション）
        migrate_existing_tables()
        
        # 全テーブルを作成（存在しない場合のみ）
        Base.metadata.create_all(bind=engine)
        print("✅ テーブル作成完了")
        
        # サンプルデータの作成
        create_sample_data()
        
    except Exception as e:
        print(f"❌ データベース初期化エラー: {e}")
        raise e

def migrate_existing_tables():
    """既存テーブルに新しいフィールドを追加するマイグレーション"""
    migration_queries = [
        # meetingsテーブルに新しいフィールドを追加
        "ALTER TABLE meetings ADD COLUMN IF NOT EXISTS description TEXT",
        "ALTER TABLE meetings ADD COLUMN IF NOT EXISTS priority VARCHAR(10)",
        "ALTER TABLE meetings ADD COLUMN IF NOT EXISTS end_time TIME",
        "ALTER TABLE meetings ADD COLUMN IF NOT EXISTS rule_violation BOOLEAN DEFAULT FALSE",
        "ALTER TABLE meetings ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        
        # meeting_modeフィールドの長さを拡張
        "ALTER TABLE meetings ALTER COLUMN meeting_mode TYPE VARCHAR(50)",
        
        # participantsテーブルのrole_typeフィールドの長さを拡張
        "ALTER TABLE participants ALTER COLUMN role_type TYPE VARCHAR(50)",
    ]
    
    with engine.connect() as conn:
        for query in migration_queries:
            try:
                conn.execute(text(query))
                print(f"✅ マイグレーション実行: {query[:50]}...")
            except Exception as e:
                print(f"⚠️  マイグレーションスキップ（既存の可能性）: {str(e)[:100]}")
        
        conn.commit()
        print("✅ データベースマイグレーション完了")

def create_sample_data():
    """サンプルデータの作成"""
    try:
        db = SessionLocal()
        
        # 組織データの存在確認と作成
        existing_orgs = db.execute(text("SELECT COUNT(*) FROM organizations")).scalar()
        if existing_orgs == 0:
            print("🏢 組織データを作成中...")
            organizations_data = [
                (1, "開発部"),
                (2, "営業部"), 
                (3, "マーケティング部"),
                (4, "人事部"),
                (5, "経理部")
            ]
            
            for org_id, org_name in organizations_data:
                db.execute(text("""
                    INSERT INTO organizations (organization_id, organization_name) 
                    VALUES (:org_id, :org_name) ON CONFLICT (organization_id) DO NOTHING
                """), {"org_id": org_id, "org_name": org_name})
            
            db.commit()
            print("✅ 組織データを作成しました")
        
        # ユーザーデータの存在確認と作成
        existing_users = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
        if existing_users == 0:
            print("👤 ユーザーデータを作成中...")
            users_data = [
                ("A000001", "管理者", "admin@example.com", "password123", 1),
                ("A000002", "田中太郎", "tanaka@example.com", "password123", 1),
                ("A000003", "佐藤花子", "sato@example.com", "password123", 2),
                ("A000004", "鈴木一郎", "suzuki@example.com", "password123", 3),
                ("A000005", "高橋美咲", "takahashi@example.com", "password123", 1),
                ("A000006", "渡辺健太", "watanabe@example.com", "password123", 2),
                ("A000007", "伊藤さくら", "ito@example.com", "password123", 4),
                ("A000008", "山田太郎", "yamada@example.com", "password123", 5)
            ]
            
            for user_id, name, email, password, org_id in users_data:
                db.execute(text("""
                    INSERT INTO users (user_id, name, email, password, organization_id) 
                    VALUES (:user_id, :name, :email, :password, :org_id) 
                    ON CONFLICT (user_id) DO NOTHING
                """), {
                    "user_id": user_id, 
                    "name": name, 
                    "email": email, 
                    "password": password, 
                    "org_id": org_id
                })
            
            db.commit()
            print("✅ ユーザーデータを作成しました")
        
        db.close()
        print("✅ サンプルデータ作成完了")
        
    except Exception as e:
        print(f"⚠️  サンプルデータ作成エラー: {e}")
        if 'db' in locals():
            db.rollback()
            db.close()

def test_database_connection():
    """データベース接続テスト"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            
            if test_value == 1:
                print("✅ データベース接続テスト成功")
                return True
            else:
                print("❌ データベース接続テスト失敗")
                return False
                
    except Exception as e:
        print(f"❌ データベース接続テストエラー: {e}")
        return False

def check_table_structure():
    """テーブル構造の確認"""
    try:
        with engine.connect() as conn:
            # meetingsテーブルの構造を確認
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'meetings'
                ORDER BY ordinal_position
            """))
            
            print("📋 meetingsテーブルの構造:")
            for row in result:
                print(f"  - {row.column_name}: {row.data_type} ({'NULL可' if row.is_nullable == 'YES' else 'NOT NULL'})")
            
            # participantsテーブルの構造を確認
            result = conn.execute(text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'participants'
                ORDER BY ordinal_position
            """))
            
            print("\n📋 participantsテーブルの構造:")
            for row in result:
                length_info = f" ({row.character_maximum_length})" if row.character_maximum_length else ""
                print(f"  - {row.column_name}: {row.data_type}{length_info}")
                
    except Exception as e:
        print(f"❌ テーブル構造確認エラー: {e}")

if __name__ == "__main__":
    print("🚀 データベース初期化を開始...")
    
    # 接続テスト
    if test_database_connection():
        # データベース初期化
        init_db()
        
        # テーブル構造確認
        check_table_structure()
        
        print("🎉 データベース初期化が完了しました！")
    else:
        print("💥 データベース接続に失敗しました")