# 回测之前需要对数据进行初始化
import time
from data_helper.downloader import download_stock_basic_from_tushare, download_cb_basic_from_tushare
from data_helper.geter import get_cb_codes_from_h5, get_stock_codes_from_h5
from data_helper.server import save_stock_basic_to_h5, save_cb_basic_to_h5

# 1. 获取所有股票列表到db
df = download_stock_basic_from_tushare()
save_stock_basic_to_h5(df)
# 2. 获取所有转债列表到db
# df = download_cb_basic_from_tushare()
# save_cb_basic_to_h5(df)
# 3. 获取所有转债的行情数据到db
from data_helper.downloader import download_cb_market_from_tushare
from data_helper.server import save_cb_market_to_h5
# 获取所有转载代码
# codes = get_cb_codes_from_h5()
# for code in codes:
#     print("正在下载转债{}的行情数据".format(code))
#     try:
#         df = download_cb_market_from_tushare(code)
#     # 超时重试
#     except Exception as e:
#         print(e)
#         time.sleep(30)
#         df = download_cb_market_from_tushare(code)
#     save_cb_market_to_h5(code, df)
#     time.sleep(0.3)
# 4. 获取所有股票的财务数据到db
from data_helper.downloader import download_stock_finance_from_tushare
from data_helper.server import save_stock_finance_to_h5
codes = get_stock_codes_from_h5()
for code in codes:
    print("正在下载股票{}的财务数据".format(code))
    try:
        df = download_stock_finance_from_tushare(code)
    # 超时重试
    except Exception as e:
        print(e)
        time.sleep(30)
        df = download_stock_finance_from_tushare(code)
    save_stock_finance_to_h5(code, df)
    time.sleep(0.3)