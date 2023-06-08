import pandas as pd


class TopFactor:
    def __init__(self, factors):
        self.factors = factors  # factors: [{"name": "factor_name", "big2small":True}}]


    def __call__(self, env):
        current_observation_df = env.current_observation_df
        current_observation_df['order_by'] = 0.0
        for factor in self.factors:
            name = factor["name"]
            big2small = not factor["big2small"]
            current_observation_df['order_by'] = current_observation_df['order_by'] + current_observation_df[name].rank(pct=True, ascending=big2small)
        self.selected_codes = current_observation_df.sort_values(by='order_by', ascending=False)['code'].values
class ConditionedWarehouse:
    """调仓"""
    def __init__(self,k=10):
        """
        :param k: 持仓数
        """
        self.k = k
    def __call__(self, env):
        # # 获取当前持仓
        positions = env.account.position
        # # 获取候选股票
        selected_codes = env.selected_codes
        # 如果当前持仓未在候选列表则卖掉
        codes_to_sell = []
        for code, position in positions.items():
            if code not in selected_codes:
                codes_to_sell.append(code)
                env.account.sell(code, position['amount'])