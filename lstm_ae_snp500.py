import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


stocks = pd.read_csv('./SP 500 Stock Prices 2014-2017.csv')
# stocks['high'] = stocks['high'].fillna("dummy")
print(stocks.shape)
daily_max = stocks['high']
df_stocks_high = stocks.pivot(index='date', columns='symbol', values='high')
google = df_stocks_high['GOOGL']
amzn = df_stocks_high['AMZN']
print("hi")
# axis = stocks['date']
# google_high = google['high']
# amazon_high = amzn['high']

plt.plot(google, label='Google')
plt.plot(amzn, label='Amazon')
plt.legend()
plt.show()