from sqlalchemy import create_engine

import os
from dotenv import load_dotenv
from pathlib import Path

# 環境変数の読み込み

base_path = Path(__file__).parents[1]
env_path = base_path / '.env'
load_dotenv(dotenv_path=env_path)

# データベース接続情報
DB_USER = os.getenv('MYSQL_USER')
DB_PASSWORD = os.getenv('MYSQL_PASSWORD')
DB_HOST = os.getenv('MYSQL_SERVER')
DB_PORT = os.getenv('MYSQL_DB_PORT')
DB_NAME = os.getenv('MYSQL_DB')

ssl_cert = str('DigiCertGlobalRootCA.crt.pem')

# MySQLのURL構築
# DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")

print("●●●nakano●●● DATABASE_URL:",DATABASE_URL)
print("●●●nakano●●● SSL_CERT_PATH:", SSL_CERT_PATH)

# エンジンの作成
engine = create_engine(
    DATABASE_URL,
    connect_args = {
        "sslmode": "verify-full",
        "sslrootcert": SSL_CERT_PATH

    #    "ssl":{
    #        "ssl_ca":ssl_cert
    #    }
    },
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600
)

