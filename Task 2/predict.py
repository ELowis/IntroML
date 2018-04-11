import numpy as np
import sklearn
from sklearn import linear_model, svm, neighbors, metrics, model_selection
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

####################################################
# Perform Cross-Validation to find best classifier #
####################################################

# Load data
csv = np.loadtxt('train.csv', delimiter=',', skiprows=1)
id = csv[:, 0]
y = csv[:, 1]
X = csv[:, 2:len(csv[0])]
n = id.size

# Constants
RIDGE = 'Ridge'
RBF = 'RBF kernel'
NN = 'Nearest Neighbors'
folds = 5

penalties = [10 ** i for i in range(-3, 4)] # 1e^-3 ... 1e^3
bandwidths = [10 ** i for i in range(-3, 4)] # 1e^-3 ... 1e^3
k_neighbors = [2*i + 1 for i in range(0, 25)] # Odd numbers from 1 to 49
classifiers = {}
errors = {}

# Returns a string describing a classifier
def format_info(name, params):
    res = name
    if name == RIDGE:
        res += ", C = " + str(params[0])
    if name == RBF:
        res += ", C = " + str(params[0]) + ", gamma = " + str(params[1])
    if name == NN:
        res += ", k = " + str(params[0])
    return res

# Build Classifiers
for C in penalties:
    # Add Ridge Classifier (One vs Rest approach)
    classifiers[RIDGE, (C,)] = linear_model.RidgeClassifier(alpha=C)

    # Add RBF SVM Classifiers (One vs Rest approach)
    for gamma in bandwidths:
        classifiers[RBF, (C, gamma)] = svm.SVC(
            C=C, 
            kernel='rbf', 
            gamma=gamma, 
            decision_function_shape='ovr')

# Add k-Nearest Neighbors classifiers
for k in k_neighbors: 
    classifiers[NN, (k,)] = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')


# Perform Cross-Validation
for key in classifiers:
    name, params = key
    print("Performing " + str(folds) + "-fold CV for " + format_info(name, params) + " ...", end='')
    classifier = classifiers[key]
    #TODO: Use RMSE or accuracy as metric?
    scores = model_selection.cross_val_score(
        classifier, 
        X, 
        y, 
        cv=folds, 
        scoring='neg_mean_squared_error')
    RMSE = (-np.mean(scores)) ** 0.5

    print("\tRMSE = " + str(RMSE))
    errors[key] = RMSE

# Find min error
best_name, best_params = min(errors, key=errors.get)
best_regr = classifiers[best_name, best_params]

print("\n" + "Best regression for the data: " + format_info(best_name, best_params))
best_regr.fit(X, y)


#############################
# Predict Class on test set #
#############################

# Load test set
csv_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)
id_test = csv_test[:, 0]
X_test = csv_test[:, 1:len(csv_test[0])]
n_test = id_test.size
assert X_test.shape[1] == X.shape[1]

# Predict
y_pred = best_regr.predict(X_test)
assert y_pred.shape == id_test.shape

# Store
id_test_ = np.array([id_test]).transpose()
y_pred_ = np.array([y_pred]).transpose()
res_array = np.concatenate((id_test_, y_pred_), axis=1)
np.savetxt(
    'res.csv', 
    res_array,
    fmt='%d', 
    delimiter=',', 
    header='Id,y', 
    comments='')