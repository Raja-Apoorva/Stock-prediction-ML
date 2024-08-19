import numpy as np
import pandas as pd
from util import *
from NN import *
from DecisionTree import *
from OMPCV import *
from lstm import *
from GradientBoost import *




symbols     = ['RUSSIAN','SHANGHAI', 'NIKKEI','NASDAQ','META','GOOG (2)','IBM (1)','MSFT','AAPL (2)','GLD (1)','SPY']
# symbols     = ['RUSSIAN', 'NIKKEI','NASDAQ','META','GOOG (2)','IBM (1)','MSFT','AAPL (2)','GLD (1)','SPY'] ## use this for data greater than one year

dates = pd.date_range('2021-11-01', '2022-11-18')
prices_all = get_data(symbols,dates)

prices_all.ffill(axis=0,inplace=True)
prices_all.bfill(axis=0,inplace=True)

daily_rets = (prices_all[1:] / prices_all[:-1].values) - 1

x = daily_rets.iloc[:,:-1]
y = daily_rets.iloc[:,-1]

y = y.to_frame()

x_train = x.iloc[0:int(len(x)*0.8),:]
y_train = y.iloc[0:int(len(x)*0.8),:]
x_test = x.iloc[int(len(x)*0.8):,:]
y_test = y.iloc[int(len(x)*0.8):,:]

##############################################################################
DecisionTree(x_train,y_train,x_test,y_test) # implements Decision Tree

##############################################################################

NN(x_train,y_train,x_test,y_test) # implements Neural Network

##############################################################################

OMPC(x_train,y_train,x_test,y_test) # implements OMPC


###############################################################################

lstm(x_train,y_train,x_test,y_test) # implements LSTM

######################################################################################

GB(x_train,y_train,x_test,y_test) # implements GB