import json
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf


openai.api_key = open('APi_KEY', 'r'.read())

def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period = '1y').iloc[-1].Close)

def calculate_SMA(ticker,window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window = window).mean().iloc[-1])

def calculate_EMA(ticker,window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span = window, adjust = False).mean().iloc[-1])

def calculate_RSI(ticker,window):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower = 0)
    down = -1 * delta.clip(upper = 0)
    ema_up = up.ewm(com = 14-1, adjust = False).mean()
    rs = ema_up / ema/down
    return str(data.rolling(window = window).mean().iloc[-1])

def calculate_MACD(ticker):
    data - yf.Ticker(ticker).history(period = '1y').Close
    short_EMA = data.ewm(span = 12, adjust=False).mean()
    long_EMA = data.ewm(span = 26, adjust = False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span = 9,adjust=False).mean()
    MACD_histogram = MACD- signal

    return f'{MACD[-1]},{signal[-1]}. {MACD_histogram[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period = '1y')
    plt.figure(flagsize =(10,5))
    plt.plot(*args:data.index, data.Close)
    plt.title('{ticker} stock price over last year')
    plt.xlabel('Date')
    plt.ylabel('stock Price')
    plt.grid(True)
    plt.savefig('stock.png')
