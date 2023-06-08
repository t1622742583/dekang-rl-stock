import re
import requests
import pandas as pd
from cachetools import cached, TTLCache

class WenCai:

    @classmethod
    def query(cls, question, columns=[], limit=100) -> pd.DataFrame:
        """
        问财查询 iwencai.com
        :return: 查询结果
        :rtype: pandas.DataFrame
        """
        if question is None or len(question) == 0:
            raise('必须定义参数question')
        if type(columns) != list:
            raise('columns必须定义为list')

        v = cls.get_ticket(cls)
        url = "http://iwencai.com/customized/chart/get-robot-data"
        payload = {
            "question": question,
            "perpage":limit,
            "page":1,
            "source":"Ths_iwencai_Xuangu",
            "version":"2.0"
        }
        r = requests.post(url, json=payload, headers=cls.__get_header(cls, v=v))
        if r.status_code != 200:
            raise Exception('问财网站连不上?')
        data_json = r.json()
        if data_json['status_code'] != 0:
            raise Exception(data_json['status_msg'])
        if len(data_json['data']['answer']) == 0:
            return None

        data = data_json['data']['answer'][0]['txt'][0]['content']['components'][0]['data']
        datas = [d for d in data['datas']]
        df = pd.DataFrame(datas)
        columns = cls.__get_columns(cls, df, columns)
        return df[columns]

    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def get_ticket(self):
        r = requests.get('http://47.242.67.95:8000/wencai-ticket')
        if r.status_code != 200:
            raise Exception('failed to get wencai ticket')
        return r.json()['data']

    def __get_header(self, v=''):
        return {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Cookie": "v=%s" % v,
            "Host": "iwencai.com",
            "Origin": "http://iwencai.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        }

    def __get_columns(self, df:pd.DataFrame, columns:list) -> list:
        columns.insert(0, '简称')
        columns.insert(0, '代码')
        columns = [column for column in df.columns for target in columns if column.find(target) >=0]
        return columns

if __name__ == '__main__':
    yaars = range(2015, 2021)
    month = 6
    code_wight = {}
    for yaar in yaars:
        df = WenCai.query(question=f'{yaar}年{month}月;按涨幅排行;电子商务', columns=['涨跌幅'])
        # 将动态字段”区间涨跌幅:前复权排名[数字-数字]“转换为”排名“
        df.rename(columns={df.columns[0]: '排名'}, inplace=True)
        # 对'排名'的值'10/5170'求小数
        df['排名'] = df['排名'].apply(lambda x: float(re.findall(r'\d+/\d+', x)[0].split('/')[0]) / float(re.findall(r'\d+/\d+', x)[0].split('/')[1]))
        for index, row in df.iterrows():
            code = row['股票代码']
            wight = row['排名']
            code_wight[code] = code_wight.get(code, 0) + wight
    # 按照权重排序
    code_wight = sorted(code_wight.items(), key=lambda x: x[1], reverse=True)
    print(code_wight)