from datetime import datetime

import pandas as pd
import numpy as np
import tushare as ts

ts.set_token('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
pro = ts.pro_api()

# 设置起止日期
start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# 获取交易日历史数据
calender = pro.trade_cal()

# 筛选出交易时间的日期
trade_dates = calender[calender.is_open == 1]['cal_date'].values.tolist()
pass
# Index(['exchange', 'cal_date', 'is_open', 'pretrade_date'], dtype='object')
