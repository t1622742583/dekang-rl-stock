from datetime import datetime
from engine.account import Account
from engine.data_utils import DataUtils
import pandas as pd
'''
优化数据加载，从hdf5里有一个all为key的数据，就是处理好各种特征的合集。
这样直接加载到内存即可。
'''


class Engine:
    def __init__(self, start='2015-01-01', end=datetime.now(), init_cash=100000):
        df_all = DataUtils.load_all()

        df_all = df_all[df_all.index >= start]
        self.df_all = df_all[df_all.index <= end]
        # print(df_all)
        # 账户
        self.acc = Account(init_cash=init_cash)
        self.status = Status(self, self.acc)

        # 从feed得到df，并从index取出日期列表
        self.dates = self.df_all.index.unique()
        self.index = 0
        self.curr_date = self.dates[0]
        # print(self.df_all)

    def run(self, algo_list, name='strategy'):
        self.algo_list = algo_list
        self.name = name

        while self.index < len(self.dates):
            # 跑到时间结束
            self._update_bar()
            self.algo_processor()
            self._move_cursor()

    def _update_bar(self):
        self.curr_date = self.dates[self.index] # 更新当前时间，
        return_se = self._get_curr_return_se(self.curr_date)
        self.acc.update_bar(self.curr_date, return_se) # 更新当前账户

    def _move_cursor(self):
        self.index += 1

    def algo_processor(self):
        context = {'engine': self, 'acc': self.acc}
        for algo in self.algo_list:
            if algo(context) is True:  # 如果algo返回True,直接不运行，本次不调仓
                return None

    def get_results_df(self):
        df = pd.DataFrame({'date': self.acc.cache_dates, 'portfolio': self.acc.cache_portfolio_mv})
        df['rate'] = df['portfolio'].pct_change()
        df['equity'] = (df['rate'] + 1).cumprod()
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        return df

    def show_results(self, benckmarks=['000300.SH']):
        """展示结果"""
        ana = Analyzer(self, benchmarks=benckmarks)
        ana.save_result()
        ana.show_results()
        ana.plot()

    def get_curr_date(self):
        return self.curr_date

    def _get_curr_return_se(self, date):
        df_bar = self.df_all.loc[date]
        if type(df_bar) is pd.Series:
            df_bar = df_bar.to_frame().T
        df_bar.set_index('code', inplace=True)
        return df_bar['rate']


class Analyzer:
    def __init__(self, engine: Engine, benchmarks=['000300.SH']):
        self.df_results = engine.get_results_df()
        self.engine = engine
        self.benchmarks = benchmarks

    def show_results(self):
        returns = self.df_results['rate']
        import empyrical

        print('累计收益：', round(empyrical.cum_returns_final(returns), 3))
        print('年化收益：', round(empyrical.annual_return(returns), 3))
        print('最大回撤：', round(empyrical.max_drawdown(returns), 3))
        print('夏普比', round(empyrical.sharpe_ratio(returns), 3))
        print('卡玛比', round(empyrical.calmar_ratio(returns), 3))
        print('omega', round(empyrical.omega_ratio(returns)), 3)

    def save_result(self):
        from engine.config import DATA_DIR_HDF5_BKT_RESULTS
        with pd.HDFStore(DATA_DIR_HDF5_BKT_RESULTS.resolve()) as store:
            store[self.engine.name] = self.df_results

    def plot(self):
        returns = []
        for s in self.benchmarks:
            df_bench = DataUtils.load_returns([s])
            se = df_bench['return']
            se.name = s
            returns.append(se)

        se_port = self.df_results['rate']
        se_port.name = 'strategy'
        returns.append(se_port)
        all_returns = pd.concat(returns, axis=1)
        all_returns.dropna(inplace=True)
        all_equity = (1 + all_returns).cumprod()

        import matplotlib.pyplot as plt
        all_equity.plot()
        plt.show()


class Status:
    def __init__(self, engine: Engine, acc: Account):
        self.engine = engine
        self.acc = acc

    def check_is_holding(self, symbol):
        if symbol in self.acc.curr_holding.keys():
            return True
        return False


if __name__ == '__main__':
    symbols = ['SPX', '000300.SH']
    symbols = ['N225', '000300.SH', 'ADX', '000905.SH', '399673.SZ', 'HSI', 'GDAXI']
    from engine.algo.algos import *

    names = []
    fields = []
    # fields += ['$close/Ref($close,20) -1']
    fields += ['Slope($close,20)']
    names += ['mom']

    fields += ['$mom>0.08']
    names += ['buy']

    fields += ['$mom<0']
    names += ['sell']

    fields += ['$mom']
    names += ['order_by']

    e = Engine(symbols, names, fields)

    print('初始总市值', e.acc.get_total_mv())
    e.run(algo_list=[
        RunWeekly(),
        # SelectBySignal(),
        SelectTopK(K=1),
        # PickTime(),
        WeightEqually()
    ])
    print('最终总市值', e.acc.get_total_mv())
    e.show_results(benckmarks=['000300.SH'])
