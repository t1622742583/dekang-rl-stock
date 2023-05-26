# 特征处理
import numpy as np
MAX_INT = np.iinfo(np.int32).max
MAX_SHARE_PRICE = 30000 # 最大股价
MAX_VOLUME = 1000e8  # 最大成交量
MAX_AMOUNT = 3e10  # 最大成交额
MAX_ACCOUNT_BALANCE = MAX_INT  # 最大账户余额
MAX_NUM_SHARES = 2147483647  # 最大持仓数量
MAX_OPEN_POSITIONS = 5  # 最大持仓数量
MAX_STEPS = 20000  # 最大步数
MAX_DAY_CHANGE = 1  # 最大日涨幅


class DealOpen:
    """开盘价"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE


class DealClose:
    """收盘价"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE


class DealHigh:
    """最高价"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE


class DealLow:
    """最低价"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE


class DealVolume:
    """成交量"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_VOLUME


class DealAmount:
    """成交额"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_AMOUNT


class DealTurnover:
    """换手率"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPctChg:
    """涨跌幅"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPctChg5:
    """5日涨跌幅"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPctChg10:
    """10日涨跌幅"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPctChg20:
    """20日涨跌幅"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPeTTM:
    """市盈率TTM"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 1e4


class DealPbMRQ:
    """市净率MRQ"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealPsTTM:
    """市销率TTM"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / 100


class DealAccountNowNetWorth:
    """当前净值"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_ACCOUNT_BALANCE


class DealAccountCash:
    """账户余额"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_ACCOUNT_BALANCE


class DealAccountMaxNetWorth:
    """最大账户价值"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_ACCOUNT_BALANCE


class DealAccountSharesHeld:
    """持有股票数量"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_NUM_SHARES


class DealAccountCostBasis:
    """成本基数"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE


# 股票类
class DealStockShares:
    """股票数量"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_NUM_SHARES


class DealStockAvgCost:
    """平均成本"""

    def __init__(self):
        pass

    def deal(self, value):
        return value / MAX_SHARE_PRICE
