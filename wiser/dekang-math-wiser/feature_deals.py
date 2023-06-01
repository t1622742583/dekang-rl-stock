# 理清工作流: 每天获取当天所有支持交易得转债,所有行情数据,给到它要用得行情数据[{"name","标准化方案"}],以及意外数据[{"name","标准化方案","获取方案"(参数:交易日期,代码,现有行情或者让那边通过code查)
# 拿到当前转债得行情后,遍历所有需求字段,传入当前当前行情,deal中判断build是否为空,为空则从行情中拿name值,进行标准化等处理
# 对于排名这种工作流如何实现?数学量化无非确定 买卖的触发情况
import pandas as pd

from data_helper.geter import get_stock_finance_from_h5


class BuildProfitToGr:
    """净利润增长率"""

    def __init__(self):
        pass

    def __call__(self, name, codes, trade_date, now_market):
        """
        :param name:  字段名
        :param codes:  当日转债代码
        :param trade_date:  交易日期
        :param now_market:  当日转债行情数据
        :return:
        """
        new_df = pd.DataFrame()
        for code in codes:
            # 获取当前转债正股代码
            stock_code = now_market.loc[now_market["code"] == code, "stock_code"].values[0]
            # 通过正股代码获取获取最近财务数据
            # TODO: 如果实盘环境则·需要通过接口获取最新财务数据
            df = get_stock_finance_from_h5(stock_code)
            # 查询最近财务数据中最近一期数据
            df = df.loc[df["trade_date"] <= trade_date]
            # 排序
            df = df.sort_values(by="trade_date", ascending=False)
            # 获取最近一期数据
            df = df.iloc[0]
            # 获取净利润增长率
            profit_to_gr = df["profit_to_gr"]
            new_df = new_df.append(pd.DataFrame({"code": [code], "name": [profit_to_gr]}))
        # 横向合并
        return new_df
