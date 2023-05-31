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

    def deal(self, codes, date, now_market):
        if self.build:
            value = self.build(self.name, codes, date, now_market)
        else:
            value = now_market[self.name]
        if self.normalization:
            value = self.normalization(value)
        return value


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
strategy_pipelines = [
    # 选股
    # 买入
    # 卖出
]
