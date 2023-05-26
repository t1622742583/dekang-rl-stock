# encoding: utf8
from loguru import logger
import pandas as pd
import abc

from ..engine_runner import Engine


class RunOnce:
    def __init__(self):
        self.done = False

    def __call__(self, context):
        done = self.done
        self.done = True
        return done


class RunPeriod:
    def __init__(self):
        self.last_date = None

    def __call__(self, target):
        engine = target['engine']
        now = engine.get_curr_date()

        last_date = self.last_date

        date_to_compare = last_date
        now = pd.Timestamp(now)
        date_to_compare = pd.Timestamp(date_to_compare)

        result = self.compare_dates(now, date_to_compare)
        self.last_date = now
        return result

    @abc.abstractmethod
    def compare_dates(self, now, date_to_compare):
        raise (NotImplementedError("RunPeriod Algo is an abstract class!"))


# https://github.com/pmorissette/bt/blob/master/bt/algos.py
class RunQuarterly(RunPeriod):

    def compare_dates(self, now, date_to_compare):
        if now.quarter != date_to_compare.quarter:
            return False
        return True


class RunWeekly(RunPeriod):

    def compare_dates(self, now, date_to_compare):
        if now.week != date_to_compare.week:
            return False
        return True


class RunMonthly(RunPeriod):

    def compare_dates(self, now, date_to_compare):
        if now.month != date_to_compare.month:
            return False
        return True


class SelectAll:
    def __call__(self, context):
        engine = context['engine']
        context['selected'] = list(engine.symbols)
        return False


class SelectBySignal:
    def __init__(self, buy_col='buy', sell_col='sell'):
        self.buy_col = buy_col
        self.sell_col = sell_col

    def __call__(self, context):
        engine = context['engine']
        features = engine.df_features

        to_buy = []
        to_sell = []
        holding = []

        curr_date = engine.get_curr_date()
        if curr_date not in features.index:
            logger.error('日期不存在{}'.format(curr_date))
            return True

        bar = features.loc[curr_date]
        if type(bar) is pd.Series:
            bar = bar.to_frame().T

        for row_index, row in bar.iterrows():
            # print(row_index, row)
            symbol = row['code']

            if row[self.buy_col]:
                to_buy.append(symbol)
            if row[self.sell_col]:
                to_sell.append(symbol)

            if engine.status.check_is_holding(symbol):
                holding.append(symbol)

        new_hold = list(set(to_buy + holding))
        for s in to_sell:
            if s in new_hold:
                new_hold.remove(s)

        context['selected'] = new_hold


def get_current_bar(context):
    engine = context['engine']
    features = engine.df_all

    curr_date = engine.get_curr_date()
    if curr_date not in features.index:
        logger.error('日期不存在{}'.format(curr_date))
        return None

    bar = features.loc[curr_date]
    if type(bar) is pd.Series:
        bar = bar.to_frame().T
    return bar


class SelectTopK_Multi:
    def __init__(self, K=20, factors=[]):
        """K 表示要选择多少只股票，factors专门用于计算每个股票的排名。"""
        self.K = K
        self.factors = factors

    def __call__(self, context):
        # context 是一个字典，它包含了当前策略运行的上下文信息。这里的 engine 是另一个对象，它处理数据和计算等功能。features 是 engine 中存储的所有股票特征的数据结构
        engine = context['engine']
        features = engine.df_all

        bar = get_current_bar(context)
        # 这行代码使用先前定义的函数 get_current_bar 获取当前最新的股票数据，并为每支股票添加一个新的名为 order_by 的列，初始值置为0.0
        bar['order_by'] = 0.0
        for f in self.factors:
            true_f = f
            asc = False
            if '-' in f:
                true_f = true_f.replace('-', '')
                asc = True

            if true_f not in bar.columns:
                logger.error('{}不在bar的列中'.format(true_f))
                continue
            # pct=True 计算百分比  ascending=False 从小到大排名(倒序)
            bar['order_by'] = bar['order_by'] + bar[true_f].rank(pct=True, ascending=asc)

        bar.sort_values('order_by', ascending=True, inplace=True)
        selected = []
        # 这个循环迭代 self.factors 中每个因子，处理每个因子的排名。如果因子中含有 "-"，则 asc 会变为 True，否则为 False。true_f 实际上就是因子去掉 "-" 后的名字。
        # 如果 true_f 不在 bar 的列中，则说明因子在当前股票池中无法使用，因此 logger.error() 抛出错误并从当前迭代中的下一个因子开始继续循环。
        # 如果因子不在被跟踪的因子列表中，则跳过此循环。
        # 对于每个因子，将其按照从小到大的顺序排名，并将其排名值累加到 order_by 列中。最后，对 bar 数据按照 order_by 值升序排序。由于此时排序完成的 bar 是一个 DataFrame 对象，因此可以使用 sort_values() 方法来完成排序。
        pre_selected = None
        if 'selected' in context:
            pre_selected = context['selected']
            del context['selected']
        # 检查 context 中是否有 selected 列，如果有，则将其存储在变量 pre_selected 中 ，并且删除 context 中的 selected 列。
        for code in list(bar.code):
            if pre_selected:
                if code in pre_selected:
                    selected.append(code)
            else:
                selected.append(code)
            if len(selected) >= self.K:
                break
        context['selected'] = selected
        # 然后，这个方法通过贪心的方式选择股票，如果这个股票已经在之前被选过了，那么就把它加入到 selected 列表中。如果 selected 列表中已经达到了要选的数量K，那么就跳出循环，然后将 selected 列表存储在 context 字典中的 selected 键下。最终结果是一个长度为 K 的 selected 列表，列出了被选择用于投资的股票的代码。



class SelectTopK:
    def __init__(self, K=1, order_by='order_by', b_ascending=False):
        self.K = K
        self.order_by = order_by
        self.b_ascending = b_ascending

    def __call__(self, context):
        engine = context['engine']
        features = engine.df_all

        if self.order_by not in features.columns:
            logger.error('排序字段{}未计算'.format(self.order_by))
            return

        bar = get_current_bar(context)
        if bar is None:
            logger.error('取不到bar')
            return True
        bar.sort_values(self.order_by, ascending=self.b_ascending, inplace=True)

        selected = []
        pre_selected = None
        if 'selected' in context:
            pre_selected = context['selected']
            del context['selected']

        # 当前全候选集
        # 按顺序往下选K个
        for code in list(bar.code):
            if pre_selected:
                if code in pre_selected:
                    selected.append(code)
            else:
                selected.append(code)
            if len(selected) >= self.K:
                break
        context['selected'] = selected


class PickTime:
    def __init__(self, benchmark='000300.SH', signal='signal'):
        self.benchmark = benchmark
        # self.buy = self.buy
        self.signal = signal

    def __call__(self, context):
        stra = context['strategy']
        extra = context['extra']
        df = extra[self.benchmark]

        if self.signal not in df.columns:
            logger.error('择时信号不存在')
            return True

        curr_date = stra.get_current_dt()
        if curr_date not in df.index:
            logger.error('日期不存在{}'.format(curr_date))
            return None

        bar = df.loc[curr_date]
        if type(bar) is pd.Series:
            bar = bar.to_frame().T

        if bar[self.signal][0]:
            logger.info('择时信号显示，平仓所有。')
            context['selected'] = []


class WeightEqually:
    def __init__(self):
        pass
    def __call__(self, context):
        selected = context["selected"]
        acc = context['acc'] # 账户
        engine = context['engine'] # 引擎
        curr_date = engine.get_curr_date() # 当前日期

        N = len(selected)
        if N > 0:
            weight = 1 / N
            weights = {}
            for symbol in selected:
                weights[symbol] = weight
            logger.info(curr_date)
            logger.info(weights)
            acc.adjust_weights(weights)
        # 等比例保持持仓
        # 计算所有被选中的股票的等权重，并将其作为字典 weights 填充到该账户的方法 adjust_weights 中，以相应地调整股票的权重。首先，它计算选中股票的数量 N，然后通过除以 N 计算每个股票的等权重 weight。
        # 最后，将每个股票的名称作为 key，将 weight 作为 value 存储在字典 weights 中，并输出当前日期和等权重。
        else:
            logger.error('selected为空，退出')
            return True
        return False


class WeightFix:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, context):
        selected = context["selected"]
        stra = context['strategy']
        N = len(selected)
        if N != len(self.weights):
            logger.error('标的个数与权重个数不等')
            return True

        for data, w in zip(selected, self.weights):
            stra.order_target_percent(data, w)

        return False


from .algo_utils import *


class WeightRP:
    def __init__(self, returns_df, method=None, half=False):
        self.returns_df = returns_df
        self.method = method
        self.half = half

    def __call__(self, context):
        N = 240

        def get_train_set(change_time, df):
            """返回训练样本数据"""
            # change_time: 调仓时间
            change_time = pd.to_datetime(change_time)
            df = df.loc[df.index < change_time]
            df = df.iloc[-N:]  # 每个调仓前240个交易日
            return df

        selected = context["selected"]  # select算子返回的需要分配仓位的 data集合
        engine = context['engine']
        acc = context['acc']

        dt = engine.get_curr_date()
        # print(dt)
        sub_df = get_train_set(dt, self.returns_df)

        one_cov_matrix = None
        if self.half:
            one_cov_matrix = calculate_half_cov_matrix(sub_df)
        else:
            one_cov_matrix = np.matrix(sub_df.cov() * N)

        # 1.计算协方差： 取调仓日 前N=240个交易日数据， one_cov_matrix = returns_df.cov()*240，return np.matrix(one_cov_matrix)

        # 2.计算RP权重
        weights = None
        if self.method and self.method == 'pca':
            weights = calculate_portfolio_weight(one_cov_matrix, risk_budget_objective=pca_risk_parity)
        else:
            weights = calculate_portfolio_weight(one_cov_matrix, risk_budget_objective=naive_risk_parity)
        # print(weights)

        # 按动量 加减分

        '''
        
        K = 10
        new_weights = []
        for data, w in zip(selected, weights):
            mom = stra.inds[data]['mom'][0]
            if mom >= 0.08:
                new_weights.append(w * K)
            elif mom < -0.0:
                new_weights.append(w / K)
            else:
                new_weights.append(w)

        new_weights = [w / sum(new_weights) for w in new_weights]
        print(weights, new_weights)
        '''

        print(weights)
        new_weights = {}

        for s, w in zip(selected, weights):
            new_weights[s] = w

        print(new_weights)
        acc.adjust_weights(new_weights)


def run_algos(context, algo_list):
    for algo in algo_list:
        if algo(context) is True:  # 如果algo返回True,直接不运行，本次不调仓
            return

    if 'selected' in context:
        del context['selected']
