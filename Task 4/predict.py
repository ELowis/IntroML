import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
#import keras

train_lbl = pd.read_hdf("train_labeled.h5", "train")
train_unlbl = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

n_train_lbl = train_lbl.index.size
cols_lbl = train_lbl.values.shape[1]        # 128 + 1
n_train_unlbl = train_unlbl.index.size
cols_unlbl = train_unlbl.values.shape[1]    # 128
y_train_lbl = train_lbl.values[:, 0].astype('int32')
X_train_lbl = train_lbl.values[:, 1:cols_lbl].astype('float32')
X_train_unlbl = train_unlbl.values[:, :].astype('float32')
X_train_all = np.concatenate((X_train_lbl, X_train_unlbl))
input_dim = X_train_all.shape[1]        # 128
num_classes = 10        # 0,...,9

n_test = test.index.size
X_test = test.values.astype('float32')
#assert(X_test.shape[1] == X_train_unlbl.shape[1] == X_train_lbl.shape[1] == input_dim)

#y_train_lbl = keras.utils.to_categorical(y_train_lbl, num_classes)

# class imbalance?
counts = np.bincount(y_train_lbl)
# print(counts)        # [866 983 903 925 818 793 917 954 928 913] -> relatively balanced
labeled_rows = X_train_lbl.shape[0]
unlabeled_rows = X_train_unlbl.shape[0]

# standardize data
# scaler = preprocessing.StandardScaler().fit(X_train_all)
# scaler.transform(X_train_all)
# scaler.transform(X_train_unlbl)
# scaler.transform(X_train_lbl)

#if __name__ == "__main__":      # for parallelism under windows

# train gaussian mixtures & perform model selection over the covariance types & no. of components
# scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
# lowest_bic = np.infty
# bic = []
# n_components_range = range(1, 7)
# cv_types = ['spherical', 'tied', 'diag', 'full']
# for cv_type in cv_types:
#     for n_components in n_components_range:
#         # Fit a Gaussian mixture with EM
#         gmm = GaussianMixture(n_components=n_components,
#                                       covariance_type=cv_type)
#         gmm.fit(X)
#         bic.append(gmm.bic(X))
#         if bic[-1] < lowest_bic:
#             lowest_bic = bic[-1]
#             best_gmm = gmm

# EM-algorithm for semi-supervised learning
# 10 components

# initialize thetas with labeled data
thetas = [[],[],[]]     # (mean,var^-1,weight) per class
# Gaussian Bayes Classifier: MLE for feature distribution + w_y
for y in range(10):
    mask = y_train_lbl == y
    # [μ_y, Σ_y, w_y]
    thetas[0].append(np.mean(X_train_lbl[mask], axis=0))          # μ_y (128-dim)
    thetas[1].append(np.linalg.inv(np.cov(X_train_lbl[mask], rowvar=False)))     # Σ_y^-1 (128x128-dim)
    thetas[2].append(counts[y]/labeled_rows)      #w_y = mean(P(z=y | x, Σ, μ)) = #y-labeled data/#labeled data

gammas = np.zeros((labeled_rows + unlabeled_rows, 10), dtype='float64')      #TODO: empty?
gammas[0:labeled_rows] = preprocessing.label_binarize(y_train_lbl, classes=range(10))     # TODO: sparse_output=True?
predicted_class = np.empty((labeled_rows+unlabeled_rows))
predicted_class[0:labeled_rows] = y_train_lbl

# https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
tol = 0.001
max_iter = 100
# n = all_rows = X_train_all.shape[0]
# for P(x)
gm = GaussianMixture(max_iter=1, n_components=10, weights_init=thetas[2], means_init=thetas[0], precisions_init=thetas[1])
gm.fit(X_train_lbl)     #needed for predict
print(gm.get_params()['means_init'] == thetas[0])

#ll_old = 0      #log-likelyhood?
for i in range(max_iter):
    print('\nIteration: ', i)
    print()

    # E-step: compute γ_y(x) = P(z=y | x, Σ, μ, w) for unlabeled data
    #gammas = np.zeros((10, unlabeled_rows))
    #for y in range(10):
        #for i in range(unlabeled_rows):      # for each unlabled x-row
        # w_y*P(X | Σ_y, μ_y) / P(X)
    gammas[labeled_rows:] = gm.predict_proba(X_train_unlbl)    # shape: unlbl samples * 10

    # M-step: MLE for μ_y, Σ_y, w_y
    for y in range(10):
        thetas[0][y] = np.average(X_train_all, axis=0, weights=gammas[:,y])  # μ_y (128-dim)
        #https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gaussian_mixture.py (_estimate_gaussian_covariances_full)
        #covariances = np.empty((input_dim, input_dim))
        diff = X_train_all - thetas[0][y]
        thetas[1][y] = np.linalg.inv(np.dot(gammas[:,y] * diff.T, diff) / sum(gammas[:,y]))        # Σ_y^-1 (128x128-dim)
        #thetas[y][1] = np.cov(X_train_lbl[mask], rowvar=False))
        thetas[2][y] = np.mean(gammas[:,y])  # w_y = mean(P(z=y | x, Σ, μ)) = #y-labeled data/#labeled data

    gm.set_params(means_init=thetas[0], precisions_init=thetas[1], weights_init=thetas[2])

    # if np.abs(ll_new - ll_old) < tol:
    #     break
    # ll_old = ll_new


y_test = gm.predict(X_test)

# Write results to file
id_res = np.array([test.index]).transpose()
y_res = np.array([y_test]).transpose()
res_array = np.concatenate((id_res, y_res), axis = 1)

np.savetxt(
    'res.csv',
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