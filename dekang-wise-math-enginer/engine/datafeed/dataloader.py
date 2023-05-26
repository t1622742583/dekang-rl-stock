# encoding:utf8
import pandas as pd
from loguru import logger

from engine.datafeed.expr.expr_mgr import ExprMgr
from engine.datafeed.datafeed_hdf5 import Hdf5DataFeed
from engine.config import DATA_DIR_HDF5_CACHE


class Dataloader:
    def __init__(self):
        self.expr = ExprMgr()
        self.feed = Hdf5DataFeed()

    def load_one_df(self, symbols, names, fields):
        dfs = self.load_dfs(symbols, names, fields)
        all = pd.concat(dfs)
        all.sort_index(ascending=True, inplace=True)
        all.dropna(inplace=True)
        self.data = all
        return all

    def load_from_cache(self, key):
        with pd.HDFStore(DATA_DIR_HDF5_CACHE.resolve()) as store:
            key = 'features'
            if '/' + key in store.keys():  # 注意判断keys需要前面加“/”
                logger.info('从缓存中加载...')
                data = store[key]
                return data
            else:
                logger.error('{}不存在'.format(key))
        return None



    def load_dfs(self, symbols, names, fields):
        dfs = []
        for code in symbols:
            # 直接在内存里加上字段，方便复用
            df = self.feed.get_df(code)
            if df is None:
                continue
            for name, field in zip(names, fields):
                exp = self.expr.get_expression(field)
                # 这里可能返回多个序列
                se = exp.load(code)
                if type(se) is pd.Series:
                    df[name] = se
                if type(se) is tuple:
                    for i in range(len(se)):
                        df[name + '_' + se[i].name] = se[i]
            df['code'] = code
            dfs.append(df)

        return dfs


if __name__ == '__main__':
    names = []
    fields = []

    # fields += ['BBands($close)']


    loader = Dataloader()
    df_returns = loader.load_returns(['N225', 'ADX', '000300.SH', '000013.SH'])
    print(df_returns)
    # loader.load_one_df(['N225'], names, fields)
