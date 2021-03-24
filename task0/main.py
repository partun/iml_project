import numpy as np
from sklearn.metrics import mean_squared_error


input_file = 'test.csv'
train_mode = False

data = np.genfromtxt(input_file, delimiter=',', skip_header=1)


ids = data[:,0]

if train_mode:
    y = data[:,1]
    y_pred = np.mean(data[:,2:], axis=1)
    err = mean_squared_error(y, y_pred) ** 0.5
    print(err)
else:
    y_pred = np.mean(data[:,1:], axis=1)


a = np.stack([ids, y_pred], axis=1)
np.savetxt("output.csv", a, delimiter=",", fmt=['%d','%r'],  header='Id,y', comments='')