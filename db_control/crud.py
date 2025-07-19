# uname() error回避
import platform
print("platform", platform.uname())

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, insert, delete, update, select ,BigInteger, Column
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import json
import pandas as pd

from db_control.connect import engine
# from db_control.mymodels import SHOUHIN
from sqlalchemy import func

def myinsert_torihiki(mymodel, values):
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    
    query = insert(mymodel).values(values)
    print("nakano myinsert query")

    try:
        # トランザクションを開始
        with session.begin():
            # データの挿入
            result = session.execute(query)
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、挿入に失敗しました")
        session.rollback()

    # セッションを閉じる
    session.close()
    return "inserted"

def insert_torimei_and_return(data_model, values):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # INSERT ... RETURNING で dtl_id も取り回す
        stmt = (
            insert(data_model)
            .values(**values)
            .returning(data_model.trd_id, data_model.dtl_id)
        )
        with session.begin():
            row = session.execute(stmt).one()
        # row は sqlalchemy.engine.Row(trd_id=…, dtl_id=…)
        return row
    finally:
        session.close()
        
def myinsert_torimei(mymodel, values):
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    
    query = insert(mymodel).values(values)
    print("nakano myinsert query")

    try:
        # トランザクションを開始
        with session.begin():
            # データの挿入
            result = session.execute(query)
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、挿入に失敗しました")
        session.rollback()

    # セッションを閉じる
    session.close()
    return "inserted"

def myselect_TRD_ID(mymodel):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # TRD_IDの最大値を取得
        max_trd_id = session.query(func.max(mymodel.TRD_ID)).scalar()
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、参照に失敗しました")
        max_trd_id = None
    finally:
        session.close()
    return max_trd_id

def myselect(mymodel, CODE):
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    code_str = str(CODE)
    query = session.query(mymodel).filter(mymodel.CODE == code_str)
    try:
        # トランザクションを開始
        with session.begin():
            result = query.all()
        # 結果をオブジェクトから辞書に変換し、リストに追加
        result_dict_list = []
        for SHOUHIN_info in result:
            result_dict_list.append({
                "PRD_ID": SHOUHIN_info.PRD_ID,
                "CODE": SHOUHIN_info.CODE,
                "NAME": SHOUHIN_info.NAME,
                "PRICE": SHOUHIN_info.PRICE,
                "PRICE_INC_TAX": SHOUHIN_info.PRICE_INC_TAX
            })
        # リストをJSONに変換
        result_json = json.dumps(result_dict_list, ensure_ascii=False)
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、挿入に失敗しました")

    # セッションを閉じる
    session.close()
    return result_json


def myselectAll(mymodel):
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    query = select(mymodel)
    try:
        # トランザクションを開始
        with session.begin():
            df = pd.read_sql_query(query, con=engine)
            result_json = df.to_json(orient='records', force_ascii=False)

    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、挿入に失敗しました")
        result_json = None

    # セッションを閉じる
    session.close()
    return result_json

