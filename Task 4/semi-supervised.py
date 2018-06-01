import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.semi_supervised import label_propagation
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from scipy import stats
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

y_train_unlbl = np.full((n_train_unlbl,), -1)
X_train_unlbl = train_unlbl.values[:, :].astype('float32')

y_train_all = np.concatenate((y_train_lbl, y_train_unlbl))
X_train_all = np.concatenate((X_train_lbl, X_train_unlbl))

input_dim = X_train_all.shape[1]        # 128
num_classes = 10        # 0,...,9

n_test = test.index.size
X_test = test.values.astype('float32')
#assert(X_test.shape[1] == X_train_unlbl.shape[1] == X_train_lbl.shape[1] == input_dim)

#y_train_lbl = keras.utils.to_categorical(y_train_lbl, num_classes)


# standardize data
scaler = preprocessing.StandardScaler().fit(X_train_all)
scaler.transform(X_train_all)
scaler.transform(X_train_unlbl)
scaler.transform(X_train_lbl)
# mean = np.mean(X_train_all, axis = 0)
# std = np.std(X_train_all, axis = 0)
#
# for row in range(n_train_lbl):
#     X_train_lbl[row] = (X_train_lbl[row] - mean) / std
# for row in range(n_train_unlbl):
#     X_train_unlbl[row] = (X_train_unlbl[row] - mean) / std
# for row in range(n_test):
#     X_test[row] = (X_test[row] - mean) / std

if __name__ == "__main__":      # for parallelism under windows
    n_components = 8

    print("Computing PCA for " + str(n_components) + " components...")
    #kpca = KernelPCA(n_components=n_components, kernel="linear", fit_inverse_transform=False, n_jobs=-1)
    kpca = PCA(n_components=n_components)
    X_kpca = kpca.fit_transform(X_train_all)
    test_kpca = kpca.fit_transform(test)
    
    print("Fitting LabelSpreading to PCA data...")
    clf = label_propagation.LabelSpreading(n_jobs = -1)
    clf.fit(X_kpca, y_train_all)

    # Write results to file
    print("Predicting class labels for PCA test data...")
    id_res = np.array([test.index]).transpose()
    y_res = np.array([clf.predict(test_kpca)]).transpose()
    res_array = np.concatenate((id_res, y_res), axis = 1)

    np.savetxt(
        'res_pca.csv',
        res_array,
        fmt='%d',
        delimiter=',',
        header='Id,y',
        comments=''
        )


# kpca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True, n_jobs=-1)
# X_kpca = kpca.fit_transform(X_train_all)
#
# zeros = y_train_lbl == 0
# ones = y_train_lbl == 1
# twos = y_train_lbl == 2
# threes = y_train_lbl == 3
# fours = y_train_lbl == 4
# fives = y_train_lbl == 5
# sixs = y_train_lbl == 6
# sevens = y_train_lbl == 7
# eights = y_train_lbl == 8
# nines = y_train_lbl == 9

#
# plt.figure()
# plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="C0",
#             s=20, edgecolor='k')
# plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="C1",
#             s=20, edgecolor='k')
# plt.title("Projection by KPCA")
# plt.xlabel("1st principal component in space induced by $\phi$")
# plt.ylabel("2nd component")
