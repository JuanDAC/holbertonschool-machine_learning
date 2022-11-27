#!/usr/bin/env python3
"""
File from numpy
"""

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


print(df.head())
print(df.tail())
