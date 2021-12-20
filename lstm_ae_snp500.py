import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

stocks = pd.read_csv('./SP 500 Stock Prices 2014-2017.csv')
df_stocks_high = stocks.pivot(index='date', columns='symbol', values='high')


def PrintGoogleAndAmazonMaximums():
    google = df_stocks_high['GOOGL'].values
    amzn = df_stocks_high['AMZN'].values
    date = df_stocks_high.index
    axis = [i for i in range(1007)]

    df = pd.DataFrame({'values': np.random.randint(0, 1000, 1007)},
                      index=pd.date_range(start='2014-01-02', end='2017-12-29', periods=1007))
    fig, ax1 = plt.subplots()
    plt.plot(df.index, google, label='Google')
    plt.plot(df.index, amzn, label='Amazon')
    monthyearFmt = mdates.DateFormatter('%Y %B')
    ax1.xaxis.set_major_formatter(monthyearFmt)
    _ = plt.xticks(rotation=90)
    plt.title('Google and Amazon daily maximums')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# PrintGoogleAndAmazonMaximums()
