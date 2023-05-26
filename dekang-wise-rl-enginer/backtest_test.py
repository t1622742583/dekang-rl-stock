import argparse
from collections import defaultdict
from typing import List

import gym
import numpy as np
import pandas as pd
import tushare as ts
from gym import spaces
from loguru import logger
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# 初始化pro接口
pro = ts.pro_api('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
STOCKS_DB = 'dbs/stocks.h5'
BASIC_DB = 'dbs/basic.h5'


def get_stock_market_from_h5(code, start_date, end_date):
    """ 从.h5中查询出当前股票该时期的行情数据"""
    with pd.HDFStore(STOCKS_DB) as store:
        df = store[code]
        df = df[df.index >= start_date]
        df = df[df.index <= end_date]
        return df


def download_stock_market_from_tushare(code, start_date, end_date):
    """ 从tushare下载当前股票该时期的行情数据"""
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df.rename(columns={'trade_date': 'date', 'ts_code': 'code', 'vol': 'volume'}, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    return df


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


def get_trade_days(start_date, end_date):
    """ 获取当前股票该时期所有交易日"""
    with pd.HDFStore(BASIC_DB) as store:
        # 查询>start_date <end_date的所有交易日
        trade_days = store['trade_days']
    if trade_days.empty:
        # 下载数据
        trade_days = pro.trade_cal(exchange='')
        # 保存到.h5
        with pd.HDFStore(BASIC_DB) as store:
            store['trade_days'] = trade_days
    trade_days = trade_days[trade_days.index >= start_date]
    trade_days = trade_days[trade_days.index <= end_date]
    trade_days = trade_days.index.tolist()

    return trade_days


# 获取最大整数
MAX_INT = np.iinfo(np.int32).max
MAX_ACCOUNT_BALANCE = MAX_INT  # 最大账户余额
MAX_NUM_SHARES = 2147483647  # 最大持仓数量
MAX_SHARE_PRICE = 5000  # 最大股价
MAX_VOLUME = 1000e8  # 最大成交量
MAX_AMOUNT = 3e10  # 最大成交额
MAX_OPEN_POSITIONS = 5  # 最大持仓数量
MAX_STEPS = 20000  # 最大步数
MAX_DAY_CHANGE = 1  # 最大日涨幅


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
        # self.curr_holding = defaultdict(float)  # 当前持仓{symbol:市值}
        #
        # self.cache_dates = []  # 日期序列
        self.cache_portfolio_mv = []  # 每日市值序列
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


class TradingEnv(gym.Env):
    """交易环境"""
    metadata = {'render.modes': ['human']}  # 人类可读的模式

    def __init__(self,
                 market_df: pd.DataFrame,  # 行情数据
                 features: List[str],  # 特征列表
                 initial_balance: float = 10000,  # 初始账户余额
                 commission: float = 0.0003,
                 stamp_duty: float = 0.001,  # 印花税 千1
                 ):
        super(TradingEnv, self).__init__()  # 父类初始化
        self.account = Account(init_cash=initial_balance, commission=commission, stamp_duty=stamp_duty)  # 账户
        self.market_df = market_df  # 行情数据
        self.initial_balance = initial_balance  # 初始账户余额
        self.commission = commission  # 手续费
        self.stamp_duty = stamp_duty  # 印花税
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # 奖励范围

        # 定义动作空间
        # 0: 买入 1: 卖出 2: 无操作

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 定义观察空间 shape=(19,)意思是一维数组，长度为19
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(features) + 4,), dtype=np.float16)

    def _next_observation(self):
        """获取下一个观察值"""
        # TODO:动态特征值
        # 困难1:对数据进行标准化 可以写好标准化方案（封装成类），字典传入 如：[{"name":"open","deal":DealOpen}]
        obs = np.array([
            self.market_df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,  # 开盘价
            self.market_df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,  # 最高价
            self.market_df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,  # 最低价
            self.market_df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,  # 收盘价
            self.market_df.loc[self.current_step, 'volume'] / MAX_VOLUME,  # 成交量
            self.market_df.loc[self.current_step, 'amount'] / MAX_AMOUNT,  # 成交额
            self.market_df.loc[self.current_step, 'adjustflag'] / 10,  # 复权状态
            self.market_df.loc[self.current_step, 'tradestatus'] / 1,  # 交易状态
            self.market_df.loc[self.current_step, 'pctChg'] / 100,  # 涨跌幅
            self.market_df.loc[self.current_step, 'peTTM'] / 1e4,  # 市盈率TTM
            self.market_df.loc[self.current_step, 'pbMRQ'] / 100,  # 市净率MRQ
            self.market_df.loc[self.current_step, 'psTTM'] / 100,  # 市销率TTM
            self.market_df.loc[self.current_step, 'pctChg'] / 1e3,  # 涨跌幅
            self.account.curr_cash / MAX_ACCOUNT_BALANCE,  # 账户余额
            self.account.max_net_worth / MAX_ACCOUNT_BALANCE,  # 最大账户价值
            self.account.shares_held / MAX_NUM_SHARES,  # 持有股票数量
            self.account.cost_basis / MAX_SHARE_PRICE,  # 成本基数
        ])
        return obs

    def _take_action(self, action):
        """ 执行动作"""
        current_price = self.market_df.loc[self.current_step, "close"]  # 当前价格

        action_type = action[0]  # 动作类型
        amount = action[1]  # 动作数量

        if action_type < 1: # 0-1之间
            # 购买 amount % 的可用余额的股票
            self.account.buy(current_price)
        elif action_type < 2: # 1-2之间
            # 卖出 amount % 的持有股票
            self.account.sell(current_price)
        elif action_type < 3: # 2-3之间
            # 无操作
            self.account.keep(current_price)

    def step(self, action):
        """下一步"""
        self._take_action(action)  # 执行动作
        done = False  # 是否结束
        self.current_step += 1  # 当前步数
        if self.current_step > len(self.market_df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # 重置当前步数(可能是重新训练)
            # done = True
        # 奖励
        reward = self.account.now_net_worth - self.initial_balance  # 奖励
        reward = 1 if reward > 0 else -2  # 奖励值
        if self.account.now_net_worth < 0:
            done = True
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self, new_df: pd.DataFrame = None):
        """重置"""
        # Reset the state of the environment to an initial state
        # self.balance = self.initial_balance  # 账户余额
        self.account = Account(self.initial_balance, self.commission, self.stamp_duty)
        if new_df:
            self.market_df = new_df
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human', close=False):
        """结束"""
        return self.account.profit


def main(opt):
    # 从.h5中查询出当前股票该时期的行情数据
    market_df = get_stock_market_from_h5(opt.code, opt.start_date, opt.end_date)
    # 解决数据不全的问题
    # 获取当前股票该时期所有交易日
    trade_days = get_trade_days(opt.start_date, opt.end_date)
    # 获取当前股票该时期所有交易日的行情数据
    if market_df.empty or len(market_df) != len(trade_days):
        # 下载数据
        market_df = download_stock_market_from_tushare(opt.code, opt.start_date, opt.end_date)
        # 保存到.h5
        save_stock_market_to_h5(opt.code, market_df)
    logger.info(market_df)
    # 按中间时间段划分训练集和测试集
    train_market_df = market_df[:int(len(market_df) * 0.8)]
    test_market_df = market_df[int(len(market_df) * 0.8):]
    # 创建环境
    env = DummyVecEnv([lambda: TradingEnv(train_market_df)])  # 创建环境
    # 创建模型
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))
    # 测试环境
    env = DummyVecEnv([lambda: TradingEnv(test_market_df)])
    obs = env.reset()
    for i in range(len(test_market_df) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        # day_profits.append(profit)
        if done:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default='SH.000001', help='股票代码')
    parser.add_argument('--start_date', type=str, default='', help='开始时间')
    parser.add_argument('--mid_date', type=str, default='', help='中间时间')
    parser.add_argument('--end_date', type=str, default='', help='结束时间')
    # 使用已有模型/重新
    parser.add_argument('--benckmarks', type=List, default=['000300.SH'], help='对比'
                                                                               '基准')
    opt = parser.parse_args()
    main(opt)
