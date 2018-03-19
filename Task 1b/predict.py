# TODO: 
# -Normalize before or after adding features?
# -Cross-validation for Ridge, Lasso, SVM, L1, L2?
# -Normalize training set only with mean, variance of training set
# -Use mean, variance of training set to normalize test set

import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, metrics

csv = np.loadtxt('train.csv', delimiter=',', skiprows=1)
id = csv[:, 0]
y = csv[:, 1]
xs = csv[:, 2:len(csv[0])]
N = id.size

# Construct feature vector
linear = xs
quadratic = np.square(xs)
exponential = np.exp(xs)
cosine = np.cos(xs)
non_constant = np.concate([linear, quadratic, exponential, cosine], axis=1)
non_constant = preprocessing.scale(non_constant)

constant = np.ones(shape=(N,1))
features = np.concatenate([non-constant, constant], axis=1)

lambdas = np.array([0.1, 1, 10, 100, 1000])
regressions = np.array([])
RMSEs = np.array([])

folds = 10
n = int(N / folds)

for l in lambdas:
    print('Performing cross-validation for lambda = ' + str(l) + ' ...')
    Y_test = np.array([])
    Y_pred = np.array([])

    for i in range(folds):
        features_test = features[n*i : n*(i+1)]
        y_test = y[n*i : n*(i+1)]
        assert features_test.shape == (n, 10)
        assert y_test.shape == (n, )

        features_train = np.concatenate([features[0 : n*i], features[n*(i+1) : N]])
        y_train = np.concatenate([y[0 : n*i], y[n*(i+1) : N]])
        assert features_train.shape == (N-n, 10)
        assert y_train.shape == (N-n, )

        # Train with Ridge regression
        ridge = linear_model.Ridge(alpha=l)
        regressions.append(ridge)
        ridge.fit(features_train, y_train)
        y_pred = ridge.predict(features_test)
        assert y_test.shape == y_pred.shape

        # Store results
        Y_test = np.append(Y_test, y_test)
        Y_pred = np.append(Y_pred, y_pred)

    # Evaluate results
    assert Y_test.shape == (N, )
    assert Y_pred.shape == (N, )
    RMSE = metrics.mean_squared_error(Y_test, Y_pred) ** 0.5
    print('\tRMSE = ' + str(RMSE))
    RMSEs = np.append(RMSEs, str(RMSE))

# Find min error
min_error = min(RMSEs)
for i in range(RMSEs.size):
    if RMSEs[i] == min_error:
        best_regr = regressions[i]

        
with open('res.csv', 'w+') as res_file:
    res = '\n'.join(RMSEs)
    res_file.write(res)

