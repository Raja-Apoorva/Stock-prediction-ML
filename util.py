import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


def get_data(symbols, dates, colname='Adj Close', dir="data"):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(os.path.join(dir, "{}.csv".format(str(symbol))), index_col='Date',
                              parse_dates=True, usecols=['Date', colname], na_values=['nan'])

        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df

def plot(pred,y_test,Title):
    pred = pd.DataFrame(pred, index=y_test.index, columns=['SPY'])
    f = plt.figure()

    ax = y_test.plot(color='red', label='Actual', figsize=(50, 20))
    pred.plot(ax=ax, color='blue', label='Predicted')

    plt.legend(['Actual', 'Predicted'])
    plt.title(Title)
    plt.show()

