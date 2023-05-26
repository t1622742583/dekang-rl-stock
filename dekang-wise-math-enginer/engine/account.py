import pandas as pd
from collections import defaultdict
from loguru import logger


class Account:
    def __init__(self, init_cash=100000.0):
        self.init_cash = init_cash  # 初始资金
        self.curr_cash = self.init_cash  # 当前现金
        self.curr_holding = defaultdict(float)  # 当前持仓{symbol:市值}

        self.cache_dates = []  # 日期序列
        self.cache_portfolio_mv = []  # 每日市值序列

    # 当日收盘合，要根据se_bar更新一次及市值，再进行交易——次日开盘交易（这里有滑点）。
    def update_bar(self, date, se_bar):

        # 所有持仓的，按收益率更新mv
        total_mv = 0.0
        # 当前已经持仓中标的，使用收盘后的收益率更新
        for s, mv in self.curr_holding.items():
            rate = 0.0
            # 这里不同市场，比如海外市场，可能不存在的，不存在变化率就是0.0， 即不变
            if s in se_bar.index:
                rate = se_bar[s]
            new_mv = mv * (1 + rate)
            self.curr_holding[s] = new_mv
            total_mv += new_mv

        self.cache_portfolio_mv.append(total_mv + self.curr_cash)
        self.cache_dates.append(date)

    # weights之和需要<=1，空仓就是cash:1，只调整curr_holding/cash两个变量
    def adjust_weights(self, weights: dict):  # weights:{symbol:weight}
        if len(weights.items()) <= 0:
            logger.info('权重长度为0，清仓')
            self._close_all()
            return

        # 当前持仓总市值 + 现金部分 = 组合市值
        total_mv = self._calc_total_holding_mv()
        total_mv += self.curr_cash

        old_pos = self.curr_holding.copy()
        self.curr_holding.clear()
        for s, w in weights.items():
            self.curr_holding[s] = total_mv * w
        self.curr_cash = total_mv - self._calc_total_holding_mv()

    # 持仓市值，不包括cash
    def _calc_total_holding_mv(self):
        total_mv = 0.0
        for s, mv in self.curr_holding.items():
            total_mv += mv
        return total_mv

    def _close_all(self):
        self.curr_cash += self._calc_total_holding_mv()
        self.curr_holding.clear()

    # == 一些接口
    def get_total_mv(self):
        return self._calc_total_holding_mv() + self.curr_cash
