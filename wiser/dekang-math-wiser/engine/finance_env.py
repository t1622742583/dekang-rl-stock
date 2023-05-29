from abc import ABC

import gym
import numpy as np
from gym import spaces
from engine.engine_runner import Engine


class FinanceEnv(Engine, gym.Env, ABC):
    def __init__(self, symbols, names, fields, features):
        Engine.__init__(self, symbols, names, fields)
        gym.Env.__init__(self)
        # 正则化，和=1，长度就是组合里的证券数量
        self.df_features = self.df_features[features]
        self.action_space = spaces.Box(low=0, high=1, shape=(len(symbols),))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(symbols), len(self.df_features.columns)), dtype=np.float64
        )

        print(self.action_space, self.observation_space)

    def reset(self):
        self.index = 0
        self.curr_date = self.dates[self.index]
        df = self.df_features.loc[self.curr_date]
        print(df.values.shape)
        return df.values

    def step(self, actions):
        done = False
        if self.index >= len(self.dates) - 1:
            done = True
            return self.state, self.reward, done, {}

        self._update_bar()
        weights = self.softmax_normalization(actions)
        wts = {s: w for s, w in zip(self.symbols, weights)}
        self.acc.adjust_weights(wts)

        df = self.df_features.loc[self.dates[self.index], :]
        self.state = df.values
        self.reward = self.acc.cache_portfolio_mv[0]

        self._move_cursor()
        return self.state, self.reward, done, {}

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import A2C
    from engine.datafeed.dataloader import Dataloader

    symbols = ['399006.SZ', '000300.SH']


    names = []
    fields = []
    features = []
    # fields += ['$close/Ref($close,20) -1']
    fields += ['Slope($close,20)']
    names += ['mom']
    features += ['mom']

    env = FinanceEnv(symbols, names, fields, features)
    check_env(env)
    model = A2C("MlpPolicy", env)
    model.learn(total_timesteps=100000)
