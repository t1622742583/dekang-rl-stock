import pandas as pd

from engine.config import DATA_DIR_HDF5_ALL, DATA_DIR_HDF5_BKT_RESULTS, DATA_DIR_HDF5_BASIC


class Logic:
    @staticmethod
    def get_index_list():
        keys = []
        with pd.HDFStore(DATA_DIR_HDF5_ALL.resolve()) as store:
            keys = [k.replace('/', '') for k in store.keys()]
        return keys

    @staticmethod
    def get_bkt_results():
        with pd.HDFStore(DATA_DIR_HDF5_BKT_RESULTS.resolve()) as store:
            keys = [k.replace('/', '') for k in store.keys()]
        return keys

    @staticmethod
    def load_etfs_basic():
        with pd.HDFStore(DATA_DIR_HDF5_BASIC.resolve()) as store:
            df = store['etfs']
            df = df[['name', 'management','fund_type','list_date','benchmark','invest_type']]
            return df


if __name__ == '__main__':
    print(Logic.load_etfs_basic())
