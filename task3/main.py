import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import sys

'''
    Hostettler Maurice, Dominic Steiner

    training data: ./train.csv
    test data: ./test.csv

    prediction stored in output.csv
'''


# fixing seeds
seed_value = 48
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def char_to_id(char):
    idno = ord(char) - 65
    if 0 <= idno and idno < 26:
        return idno
    else:
        raise UnicodeError


def convert_features(col):
    n = len(col)
    data = np.zeros((n, 4, 26))

    for i, seq in enumerate(col):
        for j, c in enumerate(seq):
            data[i, j, char_to_id(c)] = 1
    return data


train_file = './train.csv'
test_file = './test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

Y_train = train_data['Active']
X_train = convert_features(train_data['Sequence'])
X_test = convert_features(test_data['Sequence'])

early_stopping = EarlyStopping(
    monitor='val_f1',
    patience=10,
    verbose=1,
    mode='max',
    restore_best_weights=True)

model = keras.Sequential([
    Flatten(input_shape=(4, 26)),
    Dense(356, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='relu')])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[f1])

model.fit(X_train, Y_train, epochs=100, batch_size=128,
          validation_split=0.15, callbacks=[early_stopping])

predictions = model.predict(X_test)
predictions[predictions > 1] = 1

np.savetxt('output.csv', np.round(
    predictions, 0), fmt="%d", delimiter=',')
