'''
Hostettler Maurice, Dominic Steiner

Packages:
numpy, sklearn

IO:
reads input form './train.csv'
output to './output.csv'

'''
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


input_file = 'train.csv'
output_file = 'output.csv'
lambdas = [0.01, 0.1, 1, 10, 100]
k = 10

data = np.genfromtxt(input_file, delimiter=',', skip_header=1)

ids = data[:, 0]
Y = data[:, 1]
X = data[:, 2:]

k_fold_spliter = GroupKFold(n_splits=k)
errors = np.zeros(len(lambdas))

for train_fold, test_fold in k_fold_spliter.split(X, y=Y, groups=ids):
    for i, lambda_ in enumerate(lambdas):
        model = Ridge(lambda_)
        model.fit(X[train_fold], Y[train_fold])
        errors[i] += mean_squared_error(Y[test_fold],
                                        model.predict(X[test_fold])) ** 0.5

errors /= k
print(errors)


np.savetxt(output_file, errors, delimiter=",", fmt='%r', comments='')
