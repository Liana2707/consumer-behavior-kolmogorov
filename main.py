import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import hankel

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense

from sklearn.model_selection import train_test_split

from lempel_ziv import *
from huffman import *
from utils import *

data = pd.read_csv('data/transact_18_22.csv', parse_dates=[2])

new_df = pd.read_csv('output.csv', parse_dates=[2], index_col=0)

selected_columns = ['date', 'socialization', 'survival', 'money', 'self_realization', 'code']

result_df = pd.read_csv('out_1.csv', index_col=0)

lzw_depth = 56

clients = list(new_df['client'].value_counts().index)

num_clients = len(clients) - 799 # Количество клиентов (обработка одного занимает 4-5 минут)
forecast_horizons = [1, 7, 14, 28]  # Горизонты прогнозирования (в днях)
start = 799

with tf.device('/GPU:0'):
    for i in range(start, start + num_clients):
        client_transactions = new_df[new_df['client'] == clients[i]].loc[:, selected_columns]
        text = client_transactions['socialization'].values

        for forecast_horizon in forecast_horizons:
            X, y = create_dataset(text, lzw_depth, forecast_horizon)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = keras.Sequential([
                LSTM(X_train.shape[1], return_sequences=True),
                Dropout(0.3),
                LSTM(X_train.shape[1], return_sequences=False),
                Dense(X_train.shape[1], activation='tanh'),
                Dense(y_train.shape[1], activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

            predictions = model.predict(X_test)
            f1, accuracy = calculate_f1_accuracy(predictions[:, -1].round().astype(int), y_test[:, -1])
            result_df.loc[result_df['client_id'] == clients[i], f'f1_{forecast_horizon}'] = f1
        if i % 10 == 0:
            result_df.to_csv('out_1.csv')
        del model