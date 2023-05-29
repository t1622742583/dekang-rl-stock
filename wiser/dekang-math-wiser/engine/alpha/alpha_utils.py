from typing import Tuple

import pandas as pd


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> Tuple[pd.Series, pd.Series]:
    """calc_ic.

    Parameters
    ----------
    pred :
        pred
    label :
        label
    date_col :
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        ic and rank ic
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


if __name__ == '__main__':
    symbols = ['N225', '000300.SH', 'ADX', '000905.SH', '399673.SZ', 'HSI', 'GDAXI']
    from engine.datafeed.dataloader import Dataloader

    names = []
    fields = []
    names += ['roc20']
    fields += ['$close/Ref($close,20)-1']

    names += ['slope20']
    fields += ['Slope($close,20)']

    periods = [1, 5, 7, 8, 10,15, 20]
    for p in periods:
        names += ['return_{}'.format(p)]
        fields += ['Ref($close,-{})/$close-1'.format(p)]

    df = Dataloader().load_one_df(symbols, names, fields)
    df['date'] = df.index
    print(df)

    for p in periods:
        for f in ['slope20']:
            ic, ric = calc_ic(df[f], df['return_{}'.format(p)], date_col='date')
            print('未来{}天收益,因子:{},ric均值：{}, 风险调整ric:{}'.format(p, f, ric.mean(), ric.mean() / ric.std()))
