# 专门负责从h5中拿数据
import pandas as pd
import tushare as ts
pro = ts.pro_api('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
STOCKS_DB = 'dbs/stocks.h5'
BASIC_DB = 'dbs/basic.h5'

STOCKS_DB = 'dbs/stocks.h5'
BASIC_DB = 'dbs/basic.h5'
def get_trade_days(start_date, end_date):
    """ 获取当前股票该时期所有交易日"""
    with pd.HDFStore(BASIC_DB) as store:
        # 查询>start_date <end_date的所有交易日
        trade_days = store['trade_days']
    if trade_days.empty:
        # 下载数据
        trade_days = pro.trade_cal(exchange='')
        # 保存到.h5
        with pd.HDFStore(BASIC_DB) as store:
            store['trade_days'] = trade_days
    trade_days = trade_days[trade_days.index >= start_date]
    trade_days = trade_days[trade_days.index <= end_date]
    trade_days = trade_days.index.tolist()

    return trade_days
def get_stock_market_from_h5(code, start_date, end_date):
    """ 从.h5中查询出当前股票该时期的行情数据"""
    with pd.HDFStore(STOCKS_DB) as store:
        df = store[code]
        df = df[df.index >= start_date]
        df = df[df.index <= end_date]
        return df