# 获取所有财务数据
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
ts.set_token('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
pro = ts.pro_api()
# 获取交易日历史数据
# calender = pro.trade_cal()
# 筛选出交易时间的日期
# trade_dates = calender[calender.is_open == 1]['cal_date'].values.tolist()
# pass
# Index(['exchange', 'cal_date', 'is_open', 'pretrade_date'], dtype='object')
df = pro.income(ts_code='600000.SH', fields='ts_code,f_ann_date,n_income')
pass