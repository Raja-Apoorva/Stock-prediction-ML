import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from util import *


def NN(x_train,y_train,x_test,y_test):
    def Model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            # tf.keras.layers.Dense(256, activation = tf.nn.leaky_relu),
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu)
        ])
        return model
    model = Model()
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=100)
    y_pred = model.predict(x_test)
    print("MSE of Neural Network: ", mean_squared_error(y_test, y_pred))
    plot(y_pred, y_test,"Neural Network")
