import numpy as np

from sklearn import linear_model, svm, neighbors, metrics, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import laplacian_kernel
# from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
# from collections import Counter

####################################################
# Perform Cross-Validation to find best classifier #
####################################################

# Load data
csv = np.loadtxt('train.csv', delimiter=',', skiprows=1)
id = csv[:, 0]
y = csv[:, 1]
X = csv[:, 2:len(csv[0])]
n = id.size

# print(Counter(y)) -> Counter({2.0: 687, 1.0: 673, 0.0: 640}) not imbalanced

# Constants
RIDGE = 'Ridge'
RBF = 'RBF kernel'
LAP = 'Laplacian kernel'
POLY = 'Polynomial kernel'
NN = 'Nearest Neighbors'
folds = 10

penalties = [10 ** i for i in range(-3, 4)] # 1e^-3 ... 1e^3
bandwidths = [10 ** i for i in range(-5, 4)] # 1e^-5 ... 1e^3
degrees = []#3*i+1 for i in range(1, 10)]
k_neighbors = [2*i + 1 for i in range(0, 25)] # Odd numbers from 1 to 49
classifiers = {}
accuracies = {}

# Returns a string describing a classifier
def format_info(name, params):
    res = name
    if name == RIDGE:
        res += ", C = " + str(params[0])
    if name in {RBF, LAP}:
        res += ", C = " + str(params[0]) + ", gamma = " + str(params[1])
    if name == POLY:
        res += ", C = " + str(params[0]) + ", gamma = " + str(params[1]) + ", degree = " + str(params[2])
    if name == NN:
        res += ", k = " + str(params[0])
    return res


if __name__ == "__main__":      # for parallelism under windows

    # Build Classifiers

    for C in penalties:
        # Add Ridge Classifier (One vs Rest approach)
        classifiers[RIDGE, (C,)] = linear_model.RidgeClassifier(alpha=C)

        for gamma in bandwidths:
            # Add RBF SVM Classifiers (One vs One approach)
            classifiers[RBF, (C, gamma)] = svm.SVC(
                kernel='rbf',
                C=C,            # regularization -> SVM
                gamma=gamma,    # bandwidth
                decision_function_shape='ovo')

            # # Add Laplacian SVM Classifiers (One vs One approach)             # always giving Accuracy = 0.34349763744093603
            # classifiers[LAP, (C, gamma)] = svm.SVC(                           # need to turn off parallelism
            #     kernel=lambda X,Y: laplacian_kernel(X,Y, gamma),
            #     C=C,
            #     decision_function_shape='ovo')

            # Add Polynomial SVM Classifier (One vs One approach)
            for d in degrees:
                classifiers[POLY, (C, gamma, d)] = svm.SVC(
                    kernel='poly',
                    C=C,
                    gamma=gamma,
                    degree=d,
                    decision_function_shape='ovo')

    # Add k-Nearest Neighbors classifiers
    for k in k_neighbors:
        classifiers[NN, (k,)] = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')

    # Perform Cross-Validation
    for key in classifiers:
        name, params = key
        print("Performing " + str(folds) + "-fold CV for " + format_info(name, params) + " ...", end='')
        classifier = classifiers[key]

        # Add a data preprocesing step (scale to zero mean and unit variance) before classification
        classifier = make_pipeline(StandardScaler(), classifier)
        classifiers[key] = classifier #Replace classifier

        scores = model_selection.cross_val_score(
            classifier,
            X,
            y,
            cv=folds,
            scoring='accuracy',
            n_jobs=-1) # use all CPUs
        mean_acc = np.mean(scores)

        print("\tMean Accuracy = " + str(mean_acc))
        accuracies[key] = mean_acc

    # Find best classifier
    best_name, best_params = max(accuracies, key=accuracies.get)
    best_classifier = classifiers[best_name, best_params]

    print("\n" + "Best regression for the data: " + format_info(best_name, best_params)
          + "\tAccuracy: " + str(accuracies[best_name, best_params]))
    best_classifier.fit(X, y)


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
    y_pred = best_classifier.predict(X_test)
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

