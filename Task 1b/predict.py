# TODO: 
# -Cross-validation for Ridge, Lasso, SVM, L1, L2?
# -Normalize training set only with mean, variance of training set
# -Use mean, variance of training set to normalize test set

import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, metrics, model_selection
from sklearn import svm

csv = np.loadtxt('train.csv', delimiter=',', skiprows=1)
id = csv[:, 0]
y = csv[:, 1]
X = csv[:, 2:len(csv[0])]
n = id.size

# Construct feature vector
linear = X
quadratic = np.square(X)
exponential = np.exp(X)
cosine = np.cos(X)
features = np.concatenate([linear, quadratic, exponential, cosine], axis=1)

lambdas = np.array([0.1, 1, 10, 100])#, 1000])
regressions = np.array([])
errors = np.array([])

for l in lambdas:
    print('Performing cross-validation for lambda = ' + str(l) + ' ...')
    regression = svm.SVR(kernel='linear', C=l)
    # Use MSE instead of RMSE since only comparison matters
    scores = model_selection.cross_val_score(regression, features, y, cv=3, scoring='neg_mean_squared_error')
    regressions = np.append(regressions, regression)
    RMSE = (-np.mean(scores)) ** 0.5

    print('\tRMSE = ' + str(RMSE))
    errors = np.append(errors, str(RMSE))

# Find min error
min_error = min(errors)
for i in range(errors.size):
    if errors[i] == min_error:
        best_regr = regressions[i]

best_regr.fit(features, y)
        
with open('res.csv', 'w+') as res_file:
    intercept = best_regr.intercept_[0]
    l = best_regr.coef_[0].tolist()
    m = map(str, l)
    sl = list(m)
    weights = '\n'.join(sl + [str(intercept)])
    res_file.write(weights)
