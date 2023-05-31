# 保存数据
import pandas as pd

FINANCE_DB = r'C:\Users\honghe\Documents\dekang-rl-stock\dbs\fiance.h5'
STOCKS_DB = r'C:\Users\honghe\Documents\dekang-rl-stock\dbs\stocks.h5'
CB_DB = r'C:\Users\honghe\Documents\dekang-rl-stock\dbs\cbs.h5'
BASIC_DB = r'C:\Users\honghe\Documents\dekang-rl-stock\dbs\basic.h5'


# CB_BASIC_DB = r'C:\Users\honghe\Documents\dekang-rl-stock\dbs\cb_basic.h5'


def save_stock_market_to_h5(code, df):
    """ 保存当前股票该时期的行情数据到.h5"""
    with pd.HDFStore(STOCKS_DB) as store:
        # 取出原始数据
        old_df = store[code]
        # 合并数据
        df = pd.concat([old_df, df])
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        # 保存
        store[code] = df


def save_stock_basic_to_h5(df):
    """ 保存所有股票基本信息到.h5"""
    with pd.HDFStore(BASIC_DB) as store:
        store['stock'] = df


def save_cb_basic_to_h5(df):
    """ 保存所有转债基本信息到.h5"""
    with pd.HDFStore(BASIC_DB) as store:
        store['cb'] = df


def save_cb_market_to_h5(code, df):
    """ 保存当前转债该时期的行情数据到.h5"""
    with pd.HDFStore(CB_DB) as store:
        # 取出原始数据
        try:
            old_df = store[code]
        except KeyError:
            old_df = pd.DataFrame()
        # 合并数据
        df = pd.concat([old_df, df])
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        # 保存
        store[code] = df


def save_stock_finance_to_h5(code, df):
    """ 保存股票财务数据到.h5"""
    with pd.HDFStore(FINANCE_DB) as store:
        # 取出原始数据
        try:
            old_df = store[code]
        except KeyError:
            old_df = pd.DataFrame()
        # 合并数据
        df = pd.concat([old_df, df])
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        # 保存
        store[code] = df
