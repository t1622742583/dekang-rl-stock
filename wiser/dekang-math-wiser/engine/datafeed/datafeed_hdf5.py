# encoding:utf8
import datetime

import pandas as pd

from engine.config import Singleton
from engine.config import DATA_DIR_HDF5_ALL
from loguru import logger


@Singleton
class Hdf5DataFeed:
    def __init__(self, db_name='index.h5'):
        print(self.__class__.__name__, '初始化...')
        self.code_dfs = {}

    def get_df(self, code, db=None):
        if code in self.code_dfs.keys():
            return self.code_dfs[code]

        with pd.HDFStore(DATA_DIR_HDF5_ALL.resolve()) as store:
            logger.debug('从hdf5里读', code)

            df = store[code]

            df['code'] = code
        self.code_dfs[code] = df
        return df

    def get_one_df_by_codes(self, codes):
        dfs = [self.get_df(code) for code in codes]
        df_all = pd.concat(dfs, axis=0)
        df_all.dropna(inplace=True)
        df_all.sort_index(inplace=True)
        return df_all


if __name__ == '__main__':
    feed = Hdf5DataFeed()
    feed2 = Hdf5DataFeed()
    print(feed.get_df('399006.SZ'))
    df = feed.get_one_df_by_codes(['000300.SH', '000905.SH', 'SPX'])
    print(df)
