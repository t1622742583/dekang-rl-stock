# 调度者
# 1.每天定时执行某布防策略
import datetime
import requests
from rocketry import Rocketry
from rocketry.conditions.api import daily

app = Rocketry()


def get_price(code, start_date, end_date, frequency):
    # TODO: 封装
    pass


@app.task(daily.after("14:55"))
def task1():
    # 获取000001.SZ的当前日线数据
    df = get_price('000001.SZ', start_date=datetime.date.today(), end_date=datetime.date.today(), frequency='daily')
    # 调用executer 获取账户信息
    account_data = requests.post('http://127.0.0.1:8000/get_account')
    # 调用wiser
