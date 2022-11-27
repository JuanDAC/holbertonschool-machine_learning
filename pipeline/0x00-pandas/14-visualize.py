#!/usr/bin/env python3
"""
File from numpy
"""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(value=df['Close'])
df['Low'] = df['Low'].fillna(value=df['Close'])
df['Open'] = df['Open'].fillna(value=df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(value=0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(value=0)
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Date')
df = df.loc['2017':]
df = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Open': 'mean',
                           'Close': 'mean', 'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum'})

df.plot()
plt.show()
