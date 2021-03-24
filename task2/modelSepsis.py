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


def sepsis_show_balance(Y):
    neg, pos = len(Y[Y == 0]), len(Y[Y == 1])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))


class ModelSepsis:
    def __init__(self, X, y, arg=None, load_weights=False, verbose=1):
        self.early_stopping = EarlyStopping(
            monitor='val_AUC',
            verbose=verbose,
            patience=4,
            mode='max',
            restore_best_weights=True)

        self.history = History()

        self.model_path = 'modelSepsis.h5'

        # sepsis_show_balance(y)
        # train val split
        features_train, self.features_val, label_train, self.label_val = train_test_split(
            X, y, test_size=0.25)
        # balance dataset
        sepsis_cases = []
        for i in range(features_train.shape[0]):
            if label_train[i] == 1:
                sepsis_cases.append(features_train[i])
        sepsis_cases = np.stack(sepsis_cases)

        balanced_train_features = features_train
        balanced_label = label_train
        for i in range(3):
            balanced_train_features = np.vstack(
                (balanced_train_features, sepsis_cases))
            balanced_label = np.concatenate(
                (balanced_label, np.ones(shape=(sepsis_cases.shape[0],))))
        shuffle = np.arange(len(balanced_label))
        np.random.shuffle(shuffle)
        self.balanced_train_features = balanced_train_features[shuffle]
        self.balanced_label = balanced_label[shuffle]

        # sepsis_show_balance(balanced_label)

        # model definition
        self.model = tf.keras.Sequential()
        self.model.add(Conv1D(26, 3, padding='same',
                              activation='relu', input_shape=(12, 36)))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(32, 4, padding='same',
                              activation='relu'))
        self.model.add(MaxPooling1D(4))
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dropout(0))
        self.model.add(Dense(1, activation='sigmoid',))
        if load_weights:
            self.model.load_weights(self.model_path)

        self.model.compile(optimizer=Adam(0.001),
                           loss=BinaryCrossentropy(),
                           metrics=[AUC(name='AUC')])

    def train(self, verbose=1):
        self.model.fit(self.balanced_train_features, self.balanced_label,
                       validation_data=(
                           self.features_val, self.label_val),
                       epochs=25,
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
