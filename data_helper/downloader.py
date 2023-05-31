# 获取数据
import pandas as pd
import tushare as ts

pro = ts.pro_api('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')


# ------------------基本数据------------------#
def download_stock_basic_from_tushare():
    """ 从tushare下载当前股票该时期的行情数据"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    df.rename(columns={'ts_code': 'code'}, inplace=True)
    df.sort_values(by='list_date', inplace=True)
    df.set_index('code', inplace=True)
    return df


def download_cb_basic_from_tushare():
    """ 从tushare下载所有转债基本信息"""
    df = pro.cb_basic(fields="ts_code,stk_code")
    df.rename(columns={'ts_code': 'code'}, inplace=True)
    # df.sort_values(by='list_date', inplace=True)
    # 去除stk_code为空的数据
    df = df[df['stk_code'].notnull()]
    df.set_index('code', inplace=True)
    return df


# ------------------行情数据------------------#
def download_cb_market_from_tushare(code: str):
    """ 从tushare 下载转载行情数据"""
    df = pro.cb_daily(ts_code=code)
    df.rename(columns={'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)

    return df


def download_stock_market_from_tushare(code, start_date, end_date):
    """ 从tushare下载当前股票该时期的行情数据"""
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df.rename(columns={'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df


# ------------------财务数据------------------#
def download_stock_finance_from_tushare(code):
    """获取当前股票所有日期的财务数据"""
    df = pro.fina_indicator(ts_code=code)
    df.rename(columns={'ts_code': 'code'}, inplace=True)
    df.set_index('end_date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df
