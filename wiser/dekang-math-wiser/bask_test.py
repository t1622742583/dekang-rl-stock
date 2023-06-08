from typing import List

from data_helper.geter import get_trade_days
from feature_deals import BuildProfitToGr


class Feature:
    def __init__(self, name, build=None, normalization=None):
        """

        :param name: 字段名称
        :param build: 构建方案
        :param normalization: 标准化方案
        """
        self.name = name
        self.build = build
        self.normalization = normalization

    def __call__(self, date, now_market):
        codes = now_market.index.values
        if self.build:
            value = self.build(self.name, codes, date, now_market)
        else:
            value = now_market[self.name]
        if self.normalization:
            value = self.normalization(value)
        return value


class Account:
    def __init__(self,
                 init_cash: float = 10000,  # 初始资金
                 commission: float = 0.0003,  # 手续费
                 stamp_duty: float = 0.001  # 印花税
                 ):
        self.init_cash = init_cash  # 初始资金
        self.curr_cash = self.init_cash  # 当前现金
        self.max_net_worth = self.init_cash  # 最大净值
        self.now_net_worth = self.init_cash  # 当前净值
        self.cost_basis = 0.0  # 每股成本
        self.shares_held = 0  # 持有股票数量
        self.commission = commission  # 手续费
        self.stamp_duty = stamp_duty  # 印花税
        self.cache_portfolio_mv = []  # 每日市值序列
        self.position = []  # 持仓
        # 比例
        self.position_ratio = []

    @property
    def profit(self):
        """收益"""
        return self.now_net_worth - self.init_cash

    def buy(self, now_price: float, buy_ratio: float = 1.0):
        """买入"""
        # 可以购买的股票数量
        max_buy_num = int(self.curr_cash / (now_price * (1 + self.commission + self.stamp_duty)))
        # 买入数量
        buy_num = int(max_buy_num * buy_ratio)
        # 买入金额
        buy_amount = buy_num * now_price * (1 + self.commission)
        # 买入后剩余现金
        self.curr_cash -= buy_amount
        # 买入后持仓数量
        self.shares_held += buy_num
        # 买入后成本
        self.cost_basis = self.cost_basis * (1 - buy_ratio) + now_price * buy_ratio
        # 更新当前净值
        self.now_net_worth = self.curr_cash + self.shares_held * now_price
        # 更新最大净值
        self.max_net_worth = max(self.now_net_worth, self.max_net_worth)
        self.cache_portfolio_mv.append(self.now_net_worth)

    def sell(self, now_price: float, sell_ratio: float = 1.0):
        """卖出"""
        # 卖出数量
        sell_num = int(self.shares_held * sell_ratio)
        # 卖出金额
        sell_amount = sell_num * now_price * (1 + self.commission + self.stamp_duty)
        # 卖出后剩余现金
        self.curr_cash += sell_amount
        # 卖出后持仓数量
        self.shares_held -= sell_num
        # 更新当前净值
        self.now_net_worth = self.curr_cash + self.shares_held * now_price
        # 更新最大净值
        self.max_net_worth = max(self.now_net_worth, self.max_net_worth)
        self.cache_portfolio_mv.append(self.now_net_worth)

    def keep(self, now_price: float):
        """暗兵不动"""
        # 更新当前净值
        self.now_net_worth = self.curr_cash + self.shares_held * now_price
        # 更新最大净值
        self.max_net_worth = max(self.now_net_worth, self.max_net_worth)
        self.cache_portfolio_mv.append(self.now_net_worth)
    # weights之和需要<=1，空仓就是cash:1，只调整curr_holding/cash两个变量


class TradingEnv():
    """交易环境"""

    def __init__(self,
                 start_date: str,  # 开始日期
                 end_date: str,  # 结束日期
                 features: List,  # 特征列表
                 initial_balance: float = 10000,  # 初始账户余额
                 commission: float = 0.0003,
                 stamp_duty: float = 0.001,  # 印花税 千1
                 strategy_pipelines: List = None,  # 策略流水线
                 ):
        super(TradingEnv, self).__init__()  # 父类初始化
        self.account = Account(init_cash=initial_balance, commission=commission, stamp_duty=stamp_duty)  # 账户
        self.start_date = start_date  # 开始日期
        self.end_date = end_date  # 结束日期
        self.initial_balance = initial_balance  # 初始账户余额
        self.commission = commission  # 手续费
        self.stamp_duty = stamp_duty  # 印花税
        self.features = features  # 特征列表
        self.strategy_pipelines = strategy_pipelines  # 策略流水线
        self.trade_days = get_trade_days(self.start_date, self.end_date)
        self.selected_codes = []  # 选中的股票/可转债代码


    def get_current_observation(self):
        # 1.获取当前日期所有行情
        # 2.
        self.current_observation_df = self.market_df.loc[self.current_date]
        for feature in self.features:
            feature_df = feature(self.current_date, self.current_observation_df)

    def step(self):
        """执行一步"""
        # 获取当前观察值
        self.get_current_observation()
        # 执行策略流水线
        for strategy_pipeline in self.strategy_pipelines:
            strategy_pipeline(self)

    def run(self):
        """运行"""
        for trade_date in self.trade_days:
            self.current_date = trade_date
            self.step()
        # TODO:总结


features = [
    # 基本行情
    Feature("open"),
    Feature("close"),
    Feature("high"),
    Feature("low"),
    Feature("volume"),
    # 外部特征
    Feature("profit_to_gr", build=BuildProfitToGr()),
]
from strategys.cb import TopFactor, ConditionedWarehouse

strategy_pipelines = [
    # 选股
    TopFactor(
        factors=[
            {
                "name": "profit_to_gr",
                "big2smail": True,
            }
        ]
    ),
    # 调仓
    ConditionedWarehouse(k=10),
]
TradingEnv(start_date="2019-01-01", end_date="2020-01-01", features=features,
           strategy_pipelines=strategy_pipelines).run()
