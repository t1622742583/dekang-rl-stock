# 接收实盘端的请求给出交易指令
import fastapi
import uvicorn
import json
import pandas as pd
from loguru import logger
from fastapi import FastAPI
from fastapi import Request
@app.post("/get_trade_signal")
async def get_trade_signal(request: Request,data: dict):
    # 账户资产
    account = data['account']
    # 持仓数量
    position = data['position']
