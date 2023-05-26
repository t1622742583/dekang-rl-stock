# encoding:utf8
# 导入tushare
import pandas as pd
import tushare as ts
from loguru import logger

# 初始化pro接口
pro = ts.pro_api('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')


def get_etf(code, offset=0, limit=600):
    # 拉取数据
    df = pro.fund_daily(**{
        "trade_date": "",
        "start_date": "",
        "end_date": "",
        "ts_code": code,
        "limit": limit,
        "offset": offset
    }, fields=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol"
    ])

    df.rename(columns={'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}, inplace=True)
    df.set_index('date', inplace=True)
    # 拉取数据
    df_adj = pro.fund_adj(**{
        "ts_code": code,
        "trade_date": "",
        "start_date": "",
        "end_date": "",
        "offset": offset,
        "limit": limit
    }, fields=[
        "trade_date",
        "adj_factor"
    ])
    df_adj.rename(columns={'trade_date': 'date'}, inplace=True)
    df_adj.set_index('date', inplace=True)
    df = pd.concat([df, df_adj], axis=1)
    df.dropna(inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] *= df['adj_factor']
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df


def get_global_index(code):
    # 拉取数据
    df = pro.index_global(**{
        "ts_code": code,
        "trade_date": "",
        "start_date": "",
        "end_date": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "open",
        "close",
        "high",
        "low",
        "vol"
    ])
    df.rename(columns={'ts_code': 'code', 'vol': 'volume', 'trade_date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)

    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df


def get_index(code):
    # 拉取数据
    df = pro.index_daily(**{
        "ts_code": code,
        "trade_date": "",
        "start_date": "",
        "end_date": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "close",
        "open",
        "high",
        "low",
        "vol",
    ])
    df.rename(columns={'ts_code': 'code', 'vol': 'volume', 'trade_date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)

    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df


def download_symbols(symbols, b_index=False):
    for symbol in symbols:
        if not b_index:  # etf
            offset = 0
            df = get_etf(symbol, offset=offset)
            while (offset < 10000):
                offset += 600
                df_append = get_etf(symbol, offset=offset, limit=600)
                if df_append is None or len(df_append) == 0:
                    break
                print(df_append.tail())
                df = df.append(df_append)
            df.sort_index(ascending=True, inplace=True)

        else:
            if '.' in symbol:
                df = get_index(symbol)
            else:
                df = get_global_index(symbol)
        print(df)
        if df is None or len(df) == 0:
            logger.error('{}错误'.format(symbol))
            continue
        with pd.HDFStore(DATA_DIR_HDF5_ALL.resolve()) as store:
            store[symbol] = df


# ================ basic =========================
def download_etfs_basic():
    # 拉取数据
    df = pro.fund_basic(**{
        "ts_code": "",
        "market": "E",
        "update_flag": "",
        "offset": "",
        "limit": "",
        "status": "L",
        "name": ""
    }, fields=[
        "ts_code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "m_fee",
        "c_fee",
        "duration_year",
        "p_value",
        "min_amount",
        "exp_return",
        "benchmark",
        "status",
        "invest_type",
        "type",
        "trustee",
        "purc_startdate",
        "redm_startdate",
        "market"
    ])
    df.rename(columns={'ts_code': 'code'}, inplace=True)
    df.sort_values(by='list_date', inplace=True)
    df.set_index('code', inplace=True)
    return df


if __name__ == '__main__':
    get_etf(code="160105.SZ")
    # from engine.config import DATA_DIR_HDF5_ALL, DATA_DIR_HDF5_BASIC
    # basic = download_etfs_basic()
    # print(basic)
    # with pd.HDFStore(DATA_DIR_HDF5_BASIC.resolve()) as store:
    #     store['etfs'] = download_etfs_basic()

    '''
    
    etfs = ['510300.SH',  # 沪深300ETF
            '159949.SZ',  # 创业板50
            '510050.SH',  # 上证50ETF
            '159928.SZ',  # 中证消费ETF
            '510500.SH',  # 500ETF
            '159915.SZ',  # 创业板 ETF
            '512120.SH',  # 医药50ETF
            '159806.SZ',  # 新能车ETF
            '510880.SH',  # 红利ETF
            '511010.SH',  # 国债
            '510500.SH',  # 中证500
            '513520.SH',  # 日经ETF
            '513030.SH',  # 德国
            '513080.SH',  # 法国CAC
            '159920.SZ',  # 恒生
            '513100.SH',  # 纳指
            ]

    etfs = [
        '510300.SH',  # 沪深300ETF
        '511260.SH',  # 十年国债ETf
        '518880.SH',  # 黄金ETF
        '511880.SH',  # 银华日利货币ETF
        # '510500.SH',  # 中证500
        # '513520.SH',  # 日经ETF
        # '513030.SH',  # 德国
        # '513080.SH',  # 法国CAC
    ]

    index = [
        'HSI',  # 恒生
        'HKTECH',  # 恒生科技
        'SPX',  # 标普500
        'IXIC',  # 纳指
        'FCHI',  # 法国CAC40
        'GDAXI',  # 德国DAX
        'FTSE',  # 富时100
        'N225',  # 日经225

        # A股
        '000016.SH',  # 上证50
        '000300.SH',  # 沪深300
        '000905.SH',  # 中证500
        '000852.SH',  # 中证1000
        '399006.SZ',  # 创业板
        '399673.SZ',  # 创业板50
        '000688.SH',  # 科创50

        '399324.SZ',  # 深证红利
    ]

    index = ['H11001.CSI']

    download_symbols(index, b_index=True)

    with pd.HDFStore(DATA_DIR_HDF5_ALL.resolve()) as store:
        print('读数据')
        print(store['000300.SH'])
    '''
