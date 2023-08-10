import sys
import os
import json
import yfinance as yf
import pandas as pd

def get_list_ticker(file_dir):
    with open(file_dir,'r') as f:
        tickers_json = json.load(f)
    return [obj['symbol'] for obj in tickers_json]

def ticker_download(list_of_tickers, start="2021-10-01", end="2022-12-30"):
    data = yf.download(list_of_tickers,start,end)
    return data

if __name__ == '__main__':
    list_of_tickers = get_list_ticker('nasdaq_constituent.json')
    data = ticker_download(list_of_tickers)
    print(data)