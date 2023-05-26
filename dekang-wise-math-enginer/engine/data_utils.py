from engine.config import DATA_DIR_HDF5_BKT_RESULTS, DATA_DIR_HDF5_ALL
from engine.datafeed.dataloader import Dataloader
import pandas as pd


class DataUtils:
    @staticmethod
    def load_all():
        with pd.HDFStore(DATA_DIR_HDF5_ALL.resolve()) as s:
            df = s['all']
        return df

    @staticmethod
    def load_returns(symbols):  # 排成一列
        names = ['return']
        fields = ['$close/Ref($close,1)-1']
        df_returns = Dataloader().load_one_df(symbols, names, fields)
        df_returns = df_returns[['return', 'code']]
        df_returns.dropna(inplace=True)
        return df_returns

    @staticmethod
    def load_returns2(symbols):  # 每支证券是一列
        dfs = Dataloader().load_dfs(symbols, names=['return'], fields=['$close/Ref($close,1)-1'])
        returns_list = []
        for symbol, df in zip(symbols, dfs):
            return_col = df['return']
            return_col.name = symbol
            returns_list.append(return_col)

        returns_df = pd.concat(returns_list, axis=1)

        # returns = df.pivot_table(index='date', columns='code', values='return')
        returns_df.dropna(inplace=True)
        return returns_df

    @staticmethod
    def load_strategy_returns(stras):
        dfs = []
        with pd.HDFStore(DATA_DIR_HDF5_BKT_RESULTS.resolve()) as store:
            for stra in stras:
                df = store[stra]
                # print(df)
                rate = df['rate']
                rate.name = stra
                dfs.append(rate)
        all = pd.concat(dfs, axis=1)
        return all

    @staticmethod
    def calc_indicators(df_returns):
        print(df_returns.index[0], df_returns.index[-1])
        import empyrical
        accu_returns = empyrical.cum_returns_final(df_returns)
        accu_returns.name = '累计收益'
        annu_returns = empyrical.annual_return(df_returns)
        annu_returns.name = '年化收益'
        max_drawdown = empyrical.max_drawdown(df_returns)
        max_drawdown.name = '最大回撤'
        sharpe = empyrical.sharpe_ratio(df_returns)
        sharpe = pd.Series(sharpe)
        sharpe.name = '夏普比'
        max_drawdown.index = annu_returns.index

        sharpe = pd.Series(empyrical.sharpe_ratio(df_returns))
        sharpe.index = accu_returns.index
        sharpe.name = '夏普比'
        all = pd.concat([accu_returns, annu_returns, max_drawdown, sharpe], axis=1)
        print(all)
        return all


if __name__ == '__main__':
    symbols = ['N225', '000300.SH', '000013.SH']

    df_returns = DataUtils.load_strategy_returns(['RSRS择时-月度再平衡'])

    df_returns = df_returns[df_returns.index <= '2022-09-25']
    print(df_returns)
    all = DataUtils.calc_indicators(df_returns)
    print(all)
