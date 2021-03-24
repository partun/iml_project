import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

'''
piecewise linear interpolation

nan values at the start and the end of the vector use nearest neighbor
vectors containing only nan values are not changed
'''
seed_value = 46
np.random.seed(seed_value)


def pwl_interpolation(v):
    v.reshape(-1, 1)
    i = 0
    # start to first number values
    while (np.isnan(v[i])):
        i += 1
        if i >= len(v):
            # all elemetets are null
            return

    for j in range(0, i):
        v[j] = v[i]

    # linear interpolation
    low = i
    while (i < len(v)):
        if np.isnan(v[i]):
            i += 1
        else:
            high = i
            for j in range(low + 1, high):
                v[j] = v[low] + (v[high] - v[low]) * ((j - low)/(high - low))

            low = high
            i += 1

    # last number value to end
    for j in range(high + 1, len(v)):
        v[j] = v[high]


'''
nearest neighbor interpolation

vectors containing only nan values are not changed
'''


def nn_interpolation(vector):
    vector.reshape(-1, 1)
    values = []
    for i in range(vector.shape[0]):
        if not np.isnan(vector[i]):
            values.append(i)

    if values == []:
        return

    # fill lower end
    for i in range(values[0]):
        vector[i] = vector[values[0]]

    for j in range(0, len(values)-1):
        lower = values[j]
        upper = values[j+1]
        mid = (upper - lower)//2 + lower + 1
        for i in range(lower + 1, mid):
            vector[i] = vector[lower]

        for i in range(mid, upper):
            vector[i] = vector[upper]

    # fill upper end
    for i in range(values[-1] + 1, len(vector)):
        vector[i] = vector[values[-1]]


def interpolate_features(features):
    for i in tqdm(range(0, features.shape[0], 12)):
        patient = features[i:i+12, :]
        assert(patient.shape == (12, 36))

        for col in range(2, patient.shape[1]):
            pwl_interpolation(patient[:, col])


def reorder(features):
    features = features.reshape(-1, 12, 36)
    # ids | age | other features
    # np.concatenate(features[::12, 0], features[::12, 2], features[::12, 2])

    return features


class Selector:

    def __init__(self, t=28):
        self.selected = []
        self.t = t

    def gen_selection(self, data, v=False):
        selected_count = 0
        not_selected_count = 0
        for i in range(0, data.shape[0]//12):
            counter = 0
            for j in range(36):
                if np.isnan((data[::12])[i, j]):
                    counter += 1
            if counter < self.t:
                self.selected.append(i)
                selected_count += 1
            else:
                not_selected_count += 1

        if v:
            print('Total:{}\n    Selected: {}\n    Removed: {} ({:.2f}% of total)\n'.format(
                selected_count + not_selected_count, selected_count, not_selected_count, 100 * not_selected_count / (not_selected_count + selected_count)))
        return selected_count, not_selected_count

    def get_selection(self):
        return self.selected

    def apply_selection(self, data):
        data = data[self.selected]


class preprocessor:
    train_feature_file = None
    train_label_file = None
    test_features_file = None

    imputer = None
    scaler = None
    selector = None

    def __init__(self, train_features_file, train_label_file, test_features_file, normalize=True, t=28):
        self.train_feature_file = train_features_file
        self.train_label_file = train_label_file
        self.test_features_file = test_features_file

        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler(with_mean=normalize, with_std=normalize)
        self.selector = Selector(t=t)

    def get_train_features(self):
        train_features = np.genfromtxt(
            self.train_feature_file, delimiter=',', skip_header=1)

        ids = train_features[::12, 0].reshape(-1, 1)
        train_features = train_features[:, 1:]
        interpolate_features(train_features)  # interpolate
        self.selector.gen_selection(train_features, v=True)
        train_features = self.imputer.fit_transform(train_features)  # impute
        train_features = self.scaler.fit_transform(train_features)  # normalize
        train_features = reorder(train_features)  # reorder
        train_features = train_features[self.selector.get_selection()]

        return ids, train_features

    def get_test_features(self):
        test_features = np.genfromtxt(
            self.test_features_file, delimiter=',', skip_header=1)

        ids = test_features[::12, 0].reshape(-1, 1)
        test_features = test_features[:, 1:]
        interpolate_features(test_features)  # interpolate
        test_features = self.imputer.transform(test_features)  # impute
        test_features = self.scaler.transform(test_features)  # normalize
        test_features = reorder(test_features)  # reorder

        return ids, test_features

    def get_train_labels(self):
        train_labels = np.genfromtxt(
            self.train_label_file, delimiter=',', skip_header=1)

        ids = train_labels[:, 0].reshape(-1, 1)
        label_tests = train_labels[:, 1:-5]
        label_sepsis = train_labels[:, -5]
        label_vitals = train_labels[:, -4:]

        label_tests = label_tests[self.selector.get_selection()]
        label_sepsis = label_sepsis[self.selector.get_selection()]
        label_vitals = label_vitals[self.selector.get_selection()]

        return ids, label_tests, label_sepsis, label_vitals
