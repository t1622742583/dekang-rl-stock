# 保存数据
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