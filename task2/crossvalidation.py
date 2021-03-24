import preprocessor as pp
from modelTest import ModelTest
from modelSepsis import ModelSepsis
from modelVitals import ModelVitals
from tqdm import tqdm
import numpy as np

train_feature_file = 'train_features.csv'
train_label_file = 'train_labels.csv'
test_features_file = 'test_features.csv'
processor = pp.preprocessor(
    train_feature_file, train_label_file, test_features_file, normalize=True, t=29)

print('starting data parsing...')
train_ids, train_features = processor.get_train_features()
label_ids, label_tests, label_sepsis, label_vitals = processor.get_train_labels()
#test_ids, test_features = processor.get_test_features()


a0 = [2,3,4,5,6,7]
a1 = [0]
a2 = [0]
a3 = [0]
args = []

for i in a0:
    for j in a1:
        for k in a2:
            for l in a3:
                args.append((i, j, k, l))



scores = []
print('{}/{}:'.format(0, len(args)))
for k, arg in enumerate(args):
    metrics = []
    try:
        for i in tqdm(range(8)):
            #model = ModelTest(train_features, label_tests, arg=arg, verbose=0)
            model = ModelSepsis(train_features, label_sepsis, arg, verbose=0)
            #model = ModelVitals(train_features, label_vitals, arg, verbose=0)
            model.train(0)
            m = model.get_val_AUC()
            metrics.append(m)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        print("Oops! An error!")
        print("Next entry.")
        print()

    if len(metrics) > 0:
        print('{}/{}: [arg={}]    mean: {:.3f}    [{:.3f}, {:.3f}] std: {:.3f}'.format(k,
                                                                           len(args), arg, np.mean(metrics), min(metrics), max(metrics), np.std(metrics)))
        scores.append(np.mean(metrics))

best_arg, best_score = max(zip(args, scores), key=lambda x: x[1])
print('arg={} got max score with {:.3f}'.format(best_arg, best_score))
best_arg, best_score = min(zip(args, scores), key=lambda x: x[1])
print('arg={} got min score with {:.3f}'.format(best_arg, best_score))

