import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, metrics, model_selection
from sklearn import svm
import matplotlib.pyplot as plt

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

# Constants
LIN_REG = 'LinearRegression'
LASSO = 'Lasso'
RIDGE = 'Ridge'
SVR = 'SVR'
folds = 5

lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
regressions = {}
errors = {}
colors = {LIN_REG: 'grey', LASSO: 'blue', RIDGE: 'green', SVR: 'red'}

# Add classical Linear Regression
regressions[LIN_REG, None] = linear_model.LinearRegression()

for l in lambdas:
    if l > 0.01: # Lasso experiences numerical instability for small lambdas
        # Add Lasso
        regressions[LASSO, l] = linear_model.Lasso(alpha=l)

    # Add Ridge
    regressions[RIDGE, l] = linear_model.Ridge(alpha=l)

    if l < 1000: # SVR is very slow to converge for big lambdas
        # Add SVR
        regressions[SVR, l] = svm.SVR(kernel='linear', C=l)

for key in regressions:
    name, param = key
    print("Performing " + str(folds) + "-fold CV for " + name, end='')
    if param:
        print(", lambda = " + str(param), end='')
    print("...", end='')
    
    regr = regressions[key]
    scores = model_selection.cross_val_score(regr, features, y, cv=folds, scoring='neg_mean_squared_error')
    RMSE = (-np.mean(scores)) ** 0.5

    print("\tRMSE = " + str(RMSE))
    errors[key] = RMSE

# Find min error
best_name, best_param = min(errors, key=errors.get)
best_regr = regressions[best_name, best_param]

print("\n" + "Best regression for the data: " + best_name, end='')
if best_param:
    print(", lambda = " + str(best_param), end='')
print()

# Plot results
plt.xscale('log')
plt.yscale('linear')
plt.xlabel("Regularization Parameter $\lambda$")
plt.ylabel("Root Mean Square Error")
plt.title("Prediction error for " + str(folds) + "-fold validation")
plotted = set() # Used to prevent multiple legend entries
for key in regressions:
    name, param = key
    if name in plotted:
        lbl = '_nolegend_'
    else:
        lbl = name
    plotted.add(name)
    if param:
        plt.plot(param, errors[key], color=colors[name], label=lbl, marker='o')
    else:
        plt.axhline(y=errors[key], color = colors[name], label=lbl, linestyle='-')

plt.legend()   
plt.show()

best_regr.fit(features, y)

with open('res.csv', 'w+') as res_file:
    w = best_regr.coef_.tolist()
    w.append(best_regr.intercept_)
    weights = '\n'.join(list(map(str, w)))
    res_file.write(weights)

#
#    intercept = best_regr.intercept_.tolist()
#    print('int: ' + str(intercept))
#    coeffs = best_regr.coef_[0].tolist()
#    print('coeff: ' + str(coeffs))
#    w = coeffs + intercept
#    print('w' + str(w))
#    weights = '\n'.join(list(map(str, w)))
#    res_file.write(weights)
