# 获取数据
import pandas as pd
import tushare as ts

pro = ts.pro_api('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
STOCKS_DB = 'dbs/stocks.h5'
BASIC_DB = 'dbs/basic.h5'


def download_stock_market_from_tushare(code, start_date, end_date):
    """ 从tushare下载当前股票该时期的行情数据"""
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df.rename(columns={'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df
