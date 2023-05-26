from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.joinpath("data")

DATA_DIR_HDF5 = DATA_DIR.joinpath('hdf5')
DATA_DIR_HDF5_ALL = DATA_DIR_HDF5.joinpath('all.h5') # 所有
DATA_DIR_HDF5_CACHE = DATA_DIR_HDF5.joinpath('cache.h5')
DATA_DIR_HDF5_BKT_RESULTS = DATA_DIR_HDF5.joinpath('bkt_results.h5')
DATA_DIR_HDF5_BASIC = DATA_DIR_HDF5.joinpath('basic.h5')

DATA_DIR_CSV = DATA_DIR.joinpath('csv')
DATA_DIR_BKT_RESULT = DATA_DIR.joinpath('bkt_result')

dirs = [DATA_DIR, DATA_DIR_CSV, DATA_DIR_BKT_RESULT]
for dir in dirs:
    dir.mkdir(exist_ok=True, parents=True)


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwagrs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwagrs)
        return _instance[cls]

    return _singleton
