import json
import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import requests
import pandas_datareader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from selenium import webdriver
import time
import os

from sklearn import preprocessing,svm

openai.api_key = open(r"C:\Users\svukk\PycharmProjects\creditcardfraiud_ml\API_KEY",'r').read()


def predictData(stock,days):
    start = datetime(2016, 1, 1)
    end = datetime.now()
    #Outputting the Historical data into a .csv for later use
    #df = get_historical_data(stock, start,output_format='pandas')
    df = data.get_data_yahoo(stock, start, end)
    print(stock)
    print("before",df.head(1))
    # csv_name = ('Exports/' + stock + '_Export.csv')
    # df.to_csv(csv_name)

    df['prediction'] = df['Close'].shift(-1)
    print("after",df.head(1))
    df.dropna(inplace=True)
    forecast_time = int(days)
    X = np.array(df.drop(['prediction'], 1))
    Y = np.array(df['prediction'])
    X = preprocessing.scale(X)
    X_prediction = X[-forecast_time:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    print(X_train)
    #Performing the Regression on the training data
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    prediction = (clf.predict(X_prediction))
    print("Prediction",prediction)
    print("hejfhiodhviodjivd")
    print(X_prediction)

def getStockPrice(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)


def calculateSMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])


def calculateEMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


def calculateRSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])


def calculateMACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal

    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'


def plotStockPrice(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title('{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()


def getTop3Leaders():
    r = requests.get("http://conu.astuce.media:9993/api/finance/group/XNGS_NASDAQ100_zLEADERS/securities/quote")
    data = r.json()
    response = []
    for x in range(0, 3):
        response.append(data[x]['security']['ticker_code'])
    return response

def getTop3Gainers():
    r = requests.get("http://conu.astuce.media:9993/api/finance/group/XNGS_TOP_GAINERS_TREP/securities/quote")
    data = r.json()
    response = []
    for x in range(0, 3):
        response.append(data[x]['security']['ticker_code'] + ": " + str(data[x]['quote']['percent_change']) + '% change')
    return response




functions = [
    {
        'name': 'getStockPrice',
        'description': 'Gets the latest stock price given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculateSMA',
        'description': 'Calculate the simple moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the SMA'
                }
            },
            'required': ['ticker', 'window'],
        },

    },
    {
        'name': 'calculateEMA',
        'description': 'Calculate the exponential moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the EMA'
                }
            },
            'required': ['ticker', 'window'],
        },
    },
    {
        'name': 'calculateRSI',
        'description': 'Calculate the RSI for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                }
            },
            'required': ['ticker'],
        },
    },
    {
        'name': 'calculateMACD',
        'description': 'Calculate the MACD for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                },
            },
            'required': ['ticker'],
        },
    },
    {
        'name': 'plotStockPrice',
        'description': 'Plot the stock price for the last year givent the ticker symbol of a company',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'

                }
            },
            'required': ['ticker'],
        },
    }
]

availableFunctions = {
    'getStockPrice': getStockPrice,
    'calculateSMA': calculateSMA,
    'calculateEMA': calculateEMA,
    'calculateRSI': calculateRSI,
    'calculateSMACD': calculateMACD,
    'plotStockPrice': plotStockPrice,

}

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('StockSage: Stock Analysis Assistant')

user_input = st.text_input('Your input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': f'{user_input}'})

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        if response_message.get('function_call'):
            function_name = response_message['function_call']['name']
            function_args = json.loads(response_message['function_call']['arguments'])
            if function_name in ['getStockPrice', 'calculateRSI', 'calculateMACD', 'plotStockPrice']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculateSMA', 'calculateEMA']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}

            function_to_call = availableFunctions[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plotStockPrice':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append(
                    {
                        'role': 'function',
                        'name': function_name,
                        'content': function_response
                    }
                )
                second_response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0613',
                    messages=st.session_state['messages']
                )
                st.text(second_response['choices'][0]['messages']['content'])
                st.session_state['messages'].append(
                    {'role': 'assisant', 'content': second_response['choices'][0]['messages']['content']})
        else:
            st.text(response_message['content'])
            st.session_state['messages'].append({'role': 'assistant', 'content': response_message['content']})
    except Exception as e:
        raise e