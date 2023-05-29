import tushare as ts

ts.set_token('854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c')
pro = ts.pro_api()


def stock_etfs_basic():
    # 拉取数据
    df = pro.fund_basic(**{
        "ts_code": "",
        "market": "E",
        "update_flag": "",
        "offset": "",
        "limit": "",
        "status": "L",
        "name": ""
    })
    df.rename(columns={'ts_code': 'code'}, inplace=True)
    df.sort_values(by='list_date', inplace=True)
    df.set_index('code', inplace=True)
    return df


if __name__ == '__main__':
    stock_etfs_basic()
