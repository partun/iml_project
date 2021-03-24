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


class ModelTest:
    def __init__(self, X, y, arg=None, load_weights=False, verbose=1):
        self.early_stopping = EarlyStopping(
            monitor='val_AUC',
            verbose=verbose,
            patience=24,
            mode='max',
            restore_best_weights=True)

        self.history = History()

        self.model_path = 'modelTest.h5'

        self.features = X
        self.labels = y

        # model definition
        self.model = tf.keras.Sequential()
        self.model.add(Conv1D(70, 3, padding='same',
                              activation='relu', input_shape=(12, 36)))
        self.model.add(MaxPooling1D(5))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(260, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='sigmoid'))
        self.model.add(Reshape((10,)))

        if load_weights:
            self.model.load_weights(self.model_path)

        self.model.compile(optimizer=Adam(0.001),
                           loss=BinaryCrossentropy(),
                           metrics=[AUC(name='AUC', multi_label=True)])

    def train(self, verbose=1):
        self.model.fit(self.features,
                       self.labels,
                       validation_split=0.15,
                       epochs=100,
                       batch_size=32,
                       callbacks=[self.early_stopping, self.history],
                       verbose=verbose)
        self.model.save_weights(self.model_path)

    def get_val_AUC(self):
        return max(self.history.history['val_AUC'])

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return roc_auc_score(y, self.model.predict(X))
