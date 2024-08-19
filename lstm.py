import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from sklearn.metrics import mean_squared_error
from util import *

def lstm(x_tr,y_tr,x_te,y_te):
    x_train = np.asarray(x_tr)
    y_train = np.asarray(y_tr)
    x_test = np.asarray(x_te)
    y_test = np.asarray(y_te)
    x_trainn = []
    y_trainn = []
    print(x_train.shape)
    for i in range(10, (len(x_train))):
        x_trainn.append(x_train[i - 10:i])
        y_trainn.append(y_train[i])
    x_trainn, y_trainn = np.array(x_trainn), np.array(y_trainn)
    x_train_shape = np.reshape(x_trainn, (x_trainn.shape[0], x_trainn.shape[1], x_trainn.shape[2]))
    model = Sequential()
    model.add(LSTM(50, input_shape=(10, x_trainn.shape[2]), return_sequences=True))
    model.add(LSTM(200, activation='relu'))
    model.add(RepeatVector(10))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model
    model.fit(x_train_shape, y_trainn, epochs=70, batch_size=32, validation_split=0.1, verbose=1)
    x_train = x_tr
    y_train = y_tr
    x_test = x_te
    y_test = y_te


    train_last = x_train.iloc[-10:]
    full_test = pd.concat((train_last, x_test), axis=0)
    full_test = np.asarray(full_test)
    y_test = np.asarray(y_test)

    x_testt = []
    y_testt = []
    for i in range(10, (len(full_test))):
        x_testt.append(full_test[i - 10:i])
        y_testt.append(y_test[i - 10])
    x_testt = np.asarray(x_testt)
    y_testt = np.asarray(y_testt)

    y_predd = model.predict(x_testt)
    y_predd = y_predd[:, 0, :]
    mean_squared_error(y_testt, y_predd)
    print("MSE of LSTM: ", mean_squared_error(y_testt, y_predd))
    testtt = pd.DataFrame(y_testt, columns=['SPY'])
    plot(y_predd,testtt,'LSTM')

