# from configs import config
from datetime import datetime
import time
import pandas as pd
import tushare as ts

renames_dict = {'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}

# 设置token
ts.set_token('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')

# 初始化pro_api
pro = ts.pro_api()


def update_cb():
    pass


def get_cb():
    # TODO: [优化项]目前是获取当天所有交易转债 去获取每只转债数据进行存储，缺点是有些转债可能在某个时间退市或转正股了，这种就没有存储，最初有个解决方案是遍历所有交易日然后遍历所有转债存储这种一方面消耗很大，8000个交易日，第二存储h5存在问题：TypeError: Cannot serialize the column [128100.SZ]
    # TODO: because its data contents are not [string] but [mixed] object dtype 我的append的数据为：[Timestamp('2023-05-19 00:00:00') 24.8 24.81 25.19 22.5 22.5 -2.3 -9.2742, 2148195.2 51657.4724]
    all_codes = []
    # 获取所有交易日
    today = datetime.today().strftime('%Y%m%d')
    calender = pro.trade_cal()
    trade_dates = calender[calender.is_open == 1]['cal_date'].values.tolist()
    trade_dates = [date for date in trade_dates if date <= today]
    trade_date = trade_dates[0]
    now_day_cb = pro.cb_daily(trade_date=trade_date)
    ts_codes = now_day_cb["ts_code"].values.tolist()
    for ts_code in ts_codes:
        with pd.HDFStore("../../../dbs/cb.h5") as store:
            if ts_code in store:
                continue
            print("ts_code:", ts_code)
            try:
                df = pro.cb_daily(ts_code=ts_code)
            except:
                time.sleep(30)
                df = pro.cb_daily(ts_code=ts_code)
            print(df.shape)
            if df.empty:
                print("空")
                continue
            df.rename(columns=renames_dict, inplace=True)
            df.date = pd.to_datetime(df.date)
            df = df.set_index('date')
            df.sort_index(ascending=True, inplace=True)

            store[ts_code] = df
            time.sleep(1)


if __name__ == '__main__':
    get_cb()
# 110063.SH