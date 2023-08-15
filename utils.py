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