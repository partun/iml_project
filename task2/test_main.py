import preprocessor as pp
from modelTest import ModelTest
from modelSepsis import ModelSepsis
from modelVitals import ModelVitals
from zipfile import ZipFile, ZIP_DEFLATED


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping, History
import keras.backend as K
import argparse


seed_value = 49
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('-tests', dest='tests', action='store_true')
parser.add_argument('-sepsis', dest='sepsis', action='store_true')
parser.add_argument('-vitals', dest='vitals', action='store_true')
parser.add_argument('-all', dest='all', action='store_true')
parser.add_argument('-notrain', dest='notrain', action='store_true')
parser.add_argument('-nopredict', dest='nopredict', action='store_true')
parser.add_argument('-loadweights', dest='loadweights', action='store_true')
args = parser.parse_args()


def gen_output_files(ids, p_tests, p_sepsis, p_vitals, filename='output.csv'):
    header = 'pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate'
    output = np.concatenate((ids, p_tests, p_sepsis, p_vitals), axis=1)

    fmt = ['%d'] + ['%.3f'] * (output.shape[1] - 1)

    np.savetxt(filename, output, delimiter=',',
               fmt=fmt, header=header, comments='')

    ZipFile('output.zip', 'w').write('output.csv', compress_type=ZIP_DEFLATED)
    code_zip = ZipFile('task2.zip', 'w')
    code_files = ['main.py', 'modelTest.py',
                  'modelSepsis.py', 'modelVitals.py', 'preprocessor.py']
    for f in code_files:
        code_zip.write(f, compress_type=ZIP_DEFLATED)


train_feature_file = 'train_features.csv'
train_label_file = 'train_labels.csv'
test_features_file = 'test_features.csv'
processor = pp.preprocessor(
    train_feature_file, train_label_file, test_features_file, normalize=True)

print('starting data parsing...')
train_ids, train_features = processor.get_train_features()
label_ids, label_tests, label_sepsis, label_vitals = processor.get_train_labels()
test_ids, test_features = processor.get_test_features()


if args.tests or args.all:
    model_test = ModelTest(
        train_features, label_tests, load_weights=args.loadweights)
    if not args.notrain:
        model_test.train()
        print(model_test.evaluate(train_features, label_tests))

    if not args.nopredict:
        tests_score = model_test.evaluate(train_features, label_tests)
        predict_tests = model_test.predict(test_features)

if args.sepsis or args.all:
    model_sepsis = ModelSepsis(
        train_features, label_sepsis, load_weights=args.loadweights)
    if not args.notrain:
        model_sepsis.train()
        print(model_sepsis.evaluate(train_features, label_sepsis))

    if not args.nopredict:
        sepsis_score = model_sepsis.evaluate(train_features, label_sepsis)
        predict_sepsis = model_sepsis.predict(test_features)

if args.vitals or args.all:
    model_vitals = ModelVitals(
        train_features, label_vitals, load_weights=args.loadweights)
    if not args.notrain:
        model_vitals.train()
        print(model_vitals.evaluate(train_features, label_vitals))

    if not args.nopredict:
        vitals_score = model_vitals.evaluate(train_features, label_vitals)
        predict_vitals = model_vitals.predict(test_features)


if (args.tests and args.sepsis and args.vitals) or args.all:
    print('expected score:')
    print(np.mean([tests_score, sepsis_score, vitals_score]))
    gen_output_files(test_ids, predict_tests, predict_sepsis, predict_vitals)
