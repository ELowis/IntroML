import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, metrics

csv = np.loadtxt('train.csv', delimiter=',', skiprows=1)
id = csv[:, 0]
y = csv[:, 1]
xs = csv[:, 2:len(csv[0])]
N = id.size

# Preprocess data
#y = preprocessing.scale(y)
#xs = preprocessing.scale(xs)

lambdas = np.array([0.1, 1, 10, 100, 1000])
RMSEs = np.array([])

folds = 10
n = int(N / folds)

for l in lambdas:
    print('Performing cross-validation for lambda = ' + str(l) + ' ...')
    Y_test = np.array([])
    Y_pred = np.array([])

    RMSE_sum = 0

    for i in range(folds):
        xs_test = xs[n*i : n*(i+1)]
        y_test = y[n*i : n*(i+1)]
        assert xs_test.shape == (n, len(xs[0]))
        assert y_test.shape == (n, )

        xs_train = np.concatenate([xs[0 : n*i], xs[n*(i+1) : N]])
        y_train = np.concatenate([y[0 : n*i], y[n*(i+1) : N]])
        assert xs_train.shape == (N-n, len(xs[0]))
        assert y_train.shape == (N-n, )

        # Train with Ridge regression
        ridge = linear_model.Ridge(alpha=l)
        ridge.fit(xs_train, y_train)
        y_pred = ridge.predict(xs_test)
        assert y_test.shape == y_pred.shape

        # Evaluate results
        RMSE_sum = RMSE_sum + metrics.mean_squared_error(y_test, y_pred) ** 0.5

    # Compute average RMSE over all folds
    RMSE_avg = RMSE_sum / folds         # maybe more precise averaging using library function?
    print('\tRMSE = ' + str(RMSE_avg))
    RMSEs = np.append(RMSEs, str(RMSE_avg))
        
with open('res.csv', 'w+') as res_file:
    res = '\n'.join(RMSEs)
    res_file.write(res)
