# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:00:21 2019

@author: Tony.George
"""

from nsepy import get_history
from datetime import date, timedelta
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

#hardcoded stocklist -- to be changed!
stocklist = ['TATASTEEL','GAIL','ONGC','VEDL','TCS','BRITANNIA','HINDUNILVR','SUNPHARMA','CIPLA','RELIANCE','TITAN',
'ITC','WIPRO','EICHERMOT','HCLTECH','GRASIM','ASIANPAINT','ADANIPORTS','HDFC','INFY',
'HINDALCO','ULTRACEMCO','COALINDIA','BAJAJFINSV','SBIN','NTPC','DRREDDY','M&M','BHARTIARTL'
,'JSWSTEEL','LT','ZEEL','POWERGRID','BAJFINANCE']

stockProfit = pd.DataFrame()

def calcProfit_MA(symbol, sma, lma):
    symbol = symbol
    print(symbol)
    enddate = date.today()
    startdate = date.today() - timedelta(days=365)
    data = get_history(symbol=symbol, start=startdate, end=enddate)
    #print(data.head())
    # Initialize the short and long windows
    short_window = sma
    long_window = lma
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    # Set the initial capital
    initial_capital= float(50000.0)
    # Create short simple moving average over the short window
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    
    # Create long simple moving average over the long window
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    # Buy 1 share
    positions['stock'] =1*signals['signal']   
    # Initialize the portfolio with value owned   
    portfolio = positions.multiply(data['Close'], axis=0)
    # Store the difference in shares owned 
    pos_diff = positions.diff()
    # Add `holdings` to portfolio
    portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    # Add `cash` to portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()   
    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()
    
    first_buy_price = int(portfolio[portfolio.index==signals[signals.positions==1].first_valid_index()]['total'])
    last_sell_price = int(portfolio[portfolio.index==signals[signals.positions==-1].last_valid_index()]['total'])
    #
    profit = last_sell_price - first_buy_price
    return profit

for i in stocklist:
    profit = calcProfit_MA(i, 40, 100)
    stockProfit = stockProfit.append({'name':i, 'profit':profit}, ignore_index=True)
    #stockProfit['name'] = i
    #stockProfit['profit']= calcProfit_MA(i, 40, 100)

print(stockProfit)
