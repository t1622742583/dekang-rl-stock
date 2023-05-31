import argparse
from typing import List
from engine.engine_runner import Engine


def main(opt):
    # 从.h5中查询出当前股票该时期的行情数据
    # 查询可以封装 调度较多，没有相应数据就自动下载
    # 初始化 数据，账户，手续费（交易+滑点）
    engine = Engine()
    # 排名
    # 要产生哪些因子 如何，自定义 计算逻辑 {"kgh":Kgh} 财务指标的获取，
    # 策略 哪种情况买入 选股 > < 排名

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='', help='开始时间')
    parser.add_argument('--end_date', type=str, default='', help='结束时间')
    parser.add_argument('--init_cash', type=int, default=100000, help='初始资金')
    # 使用已有模型/重新
    parser.add_argument('--benckmarks', type=List, default=['000300.SH'], help='对比'
                                                                               '基准')
    opt = parser.parse_args()
    main(opt)
