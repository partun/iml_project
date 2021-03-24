import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras.backend as K


def R2(y, y_pred):
    return np.mean([0.5 + 0.5 * max(0, r2_score(y[:, i], y_pred[:, i])) for i in range(4)])


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def R2_metric(y, y_pred):
    score = tf.convert_to_tensor(0.0, dtype=tf.float32)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    a = tf.constant(0.5, shape=[1], dtype=tf.float32)
    b = tf.constant(0.25, shape=[1], dtype=tf.float32)
    one = tf.constant(1.0, shape=[1], dtype=tf.float32)

    for j in range(4):
        SS_res = K.sum(K.square(y[:, j]-y_pred[:, j]))
        SS_tot = K.sum(K.square(y[:, j] - K.mean(y[:, j])))
        score = score + a + \
            (a*K.max(zero, (one - SS_res/(SS_tot + K.epsilon()))))

    return score * b


class ModelVitals:
    def __init__(self, X, y, arg=None, load_weights=False, verbose=1):
        self.early_stopping = EarlyStopping(
            monitor='val_mse',
            verbose=verbose,
            patience=24,
            mode='min',
            restore_best_weights=True)

        self.history = History()

        self.model_path = 'modelVitals.h5'

        self.features = X
        self.labels = y

        # model definition
        self.model = tf.keras.Sequential()
        self.model.add(Conv1D(40, 2, padding='same',
                              activation='relu', input_shape=(12, 36)))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(60, 3, padding='same', activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(3))
        self.model.add(Flatten())
        self.model.add(Dropout(0))
        self.model.add(Dense(250, activation='relu'))
        self.model.add(Dropout(0))
        self.model.add(Dense(4))
        self.model.add(Reshape((4,)))

        if load_weights:
            self.model.load_weights(self.model_path)

        self.model.compile(optimizer=Adam(0.001),
                           loss='mse',
                           metrics=['mse'])

    def train(self, verbose=1):
        self.model.fit(self.features,
                       self.labels,
                       validation_split=0.15,
                       epochs=200,
                       initial_epoch=0,
                       batch_size=32,
                       callbacks=[self.early_stopping, self.history],
                       verbose=verbose)
        self.model.save_weights(self.model_path)

    def get_val_AUC(self):
        return min(self.history.history['val_mse'])

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return R2(y, self.model.predict(X))
