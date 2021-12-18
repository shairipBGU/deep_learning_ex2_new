import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


def PtintGoogleAndAmazonMaximums():
    stocks = pd.read_csv('./SP 500 Stock Prices 2014-2017.csv')
    print(stocks.shape)
    daily_max = stocks['high']
    df_stocks_high = stocks.pivot(index='date', columns='symbol', values='high')
    google = df_stocks_high['GOOGL']
    amzn = df_stocks_high['AMZN']
    print("hi")
    axis = [i for i in range(1007)]
    plt.plot(axis, google, label='Google')
    plt.plot(axis, amzn, label='Amazon')
    plt.xlabel('Days from 2/1/2014')
    plt.ylabel('Price')
    plt.title('Google and Amazon daily maximums')
    plt.legend()
    plt.show()


