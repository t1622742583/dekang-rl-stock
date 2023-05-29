# 获取所有股票信息
import tushare as ts
ts.set_token('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
pro = ts.pro_api()
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
