import numpy as np
import pandas as pd
#import keras

train_lbl = pd.read_hdf("train_labeled.h5", "train")
train_unlbl = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

n_train_lbl = train_lbl.index.size
cols_lbl = train_lbl.values.shape[1]        # 128 + 1
n_train_unlbl = train_unlbl.index.size
cols_unlbl = train_unlbl.values.shape[1]    # 128
y_train_lbl = train_lbl.values[:, 0]
X_train_lbl = train_lbl.values[:, 1:cols_lbl].astype('float32')
X_train_unlbl = train_unlbl.values[:, :].astype('float32')
X_train_all = np.concatenate((X_train_lbl, X_train_unlbl))
input_dim = X_train_all.shape[1]        # 128
num_classes = 10        # 0,...,9

n_test = test.index.size
X_test = test.values.astype('float32')
#assert(X_test.shape[1] == X_train_unlbl.shape[1] == X_train_lbl.shape[1] == input_dim)

#y_train_lbl = keras.utils.to_categorical(y_train_lbl, num_classes)

# standardize data
mean = np.mean(X_train_all, axis = 0)
std = np.std(X_train_all, axis = 0)

for row in range(n_train_lbl):
    X_train_lbl[row] = (X_train_lbl[row] - mean) / std
for row in range(n_train_unlbl):
    X_train_unlbl[row] = (X_train_unlbl[row] - mean) / std
for row in range(n_test):
    X_test[row] = (X_test[row] - mean) / std


# # Write results to file
# id_res = np.array([test.index]).transpose()
# y_res = np.array([y_test]).transpose()
# res_array = np.concatenate((id_res, y_res), axis = 1)
#
# np.savetxt(
#     'res.csv',
#     res_array,
#     fmt='%d',
#     delimiter=',',
#     header='Id,y',
#     comments=''
#     )