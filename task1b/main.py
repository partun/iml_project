'''
Hostettler Maurice, Dominic Steiner

Packages:
numpy, sklearn

IO:
reads input form './train.csv'
output to './output.csv'

'''
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error

np.random.seed(10)


input_file = 'train.csv'
output_file = 'output.csv'

data = np.genfromtxt(input_file, delimiter=',', skip_header=1)

ids = data[:, 0]  # ids in first colown
Y = data[:, 1]  # Y in second colown
X = data[:, 2:]  # X features

n = X.shape[0]

# functions to call on features
functions = [np.square, np.exp, np.cos]

extended_X = X
for fn in functions:
    extended_X = np.concatenate((extended_X, fn(X)), axis=1)
# adding constant factor
extended_X = np.concatenate((extended_X, np.ones((n, 1))), axis=1)

alphas = [10**x for x in range(-7, 5)]
model = RidgeCV(alphas=alphas)
model.fit(extended_X, Y)

upper_bound = model.alpha_*10
lower_bound = model.alpha_/10

#print(lower_bound, upper_bound, model.alpha_)

# iteratively refine alpha parameter
while(upper_bound - lower_bound > 0.1):
    step = (upper_bound - lower_bound) / 10
    model = RidgeCV(np.arange(lower_bound, upper_bound, step))
    model.fit(extended_X, Y)

    upper_bound = model.alpha_ + step
    lower_bound = model.alpha_ - step

    #print(lower_bound, upper_bound, model.alpha_)


print(model.alpha_)
print(model.coef_)

# write to output.csv file
np.savetxt(output_file, model.coef_, delimiter=",", fmt='%r', comments='')
