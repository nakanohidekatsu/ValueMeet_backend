# db_control/connect.py

from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

# ========= .env の読み込み =========
# 既存：リポジトリ直下の .env を読む
base_path = Path(__file__).parents[1]
env_path = base_path / ".env"
# 追記：ファイルが無い場合でも環境変数だけは読む（安全フォールバック）
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# ========= 接続情報の取得 =========
# 追記：まずは DATABASE_URL を優先（サービスが直接発行する想定に対応）
DATABASE_URL = os.getenv("DATABASE_URL")

# 既存：分割変数（MYSQL_*）を読む
DB_USER = os.getenv("MYSQL_USER")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD")
DB_HOST = os.getenv("MYSQL_SERVER")
DB_PORT = os.getenv("MYSQL_DB_PORT")
DB_NAME = os.getenv("MYSQL_DB")

SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")  # 例：Supabase では省略可

# 追記：DATABASE_URL が無い場合は MYSQL_* から安全に組み立て
if not DATABASE_URL:
    # 不足があれば分かりやすくエラー
    missing = [k for k, v in {
        "MYSQL_USER": DB_USER,
        "MYSQL_PASSWORD": DB_PASSWORD,
        "MYSQL_SERVER": DB_HOST,
        "MYSQL_DB_PORT": DB_PORT,
        "MYSQL_DB": DB_NAME,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"DB接続情報が不足しています。{', '.join(missing)} を .env か環境変数に設定してください。"
        )

    # ユーザー名やパスワードに記号が入っても安全に
    user = DB_USER
    password = quote_plus(DB_PASSWORD)

    # 追記：Supabase想定。証明書が無い場合は sslmode=require をURL側に付与
    #       （psycopg2 は connect_args でも渡せるが、URL側に付けると分かりやすい）
    DATABASE_URL = (
        f"postgresql+psycopg2://{user}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"{'' if SSL_CERT_PATH else '?sslmode=require'}"
    )

# デバッグ出力（必要ならコメントアウト可）
print("●●●nakano●●● DATABASE_URL:", DATABASE_URL)
print("●●●nakano●●● SSL_CERT_PATH:", SSL_CERT_PATH)

# ← この行を追加
os.environ["DATABASE_URL"] = DATABASE_URL  # 他モジュール用に環境変数へ反映

# ========= SQLAlchemy エンジン作成 =========
# 追記：証明書がある場合は verify-full + sslrootcert、
#       無い場合は require のみ（Supabase推奨）
if SSL_CERT_PATH:
    connect_args = {
        "sslmode": "verify-full",
        "sslrootcert": SSL_CERT_PATH,
    }
else:
    # URLに ?sslmode=require を付けていない場合の保険
    connect_args = {"sslmode": "require"}

from sqlalchemy import create_engine  # ●●● nakano 追加：SQLクエリをログ出力
from sqlalchemy.orm import sessionmaker  # ●●● nakano 追加：SQLクエリをログ出力
from .settings import DEBUG_SQL  # ●●● nakano 追加：SQLクエリをログ出力
from db_control.crud import DEBUG_SQL  # ●●● nakano 追加：SQLクエリをログ出力
from . import crud  # ← crud → connect の参照が無ければ循環にならない
engine = create_engine(
    DATABASE_URL,
    echo=DEBUG_SQL,  # ●●● nakano 追加：SQLクエリをログ出力
    connect_args=connect_args,
#    echo=True, # ●●● nakano 追加：SQLクエリをログ出力
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # ●●● nakano 追加：SQLクエリをログ出力
crud.SessionLocal.configure(bind=engine)
# # db_control/connect.py
# from sqlalchemy import create_engine

# import os
# from dotenv import load_dotenv
# from pathlib import Path

# # 環境変数の読み込み

# base_path = Path(__file__).parents[1]
# env_path = base_path / '.env'
# load_dotenv(dotenv_path=env_path)

# # データベース接続情報
# DB_USER = os.getenv('MYSQL_USER')
# DB_PASSWORD = os.getenv('MYSQL_PASSWORD')
# DB_HOST = os.getenv('MYSQL_SERVER')
# DB_PORT = os.getenv('MYSQL_DB_PORT')
# DB_NAME = os.getenv('MYSQL_DB')

# ssl_cert = str('DigiCertGlobalRootCA.crt.pem')

# # MySQLのURL構築
# # DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")

# print("●●●nakano●●● DATABASE_URL:",DATABASE_URL)
# print("●●●nakano●●● SSL_CERT_PATH:", SSL_CERT_PATH)

# # エンジンの作成
# engine = create_engine(
#     DATABASE_URL,
#     connect_args = {
#         "sslmode": "verify-full",
#         "sslrootcert": SSL_CERT_PATH

#     #    "ssl":{
#     #        "ssl_ca":ssl_cert
#     #    }
#     },
#     echo=True,
#     pool_pre_ping=True,
#     pool_recycle=3600
# )

