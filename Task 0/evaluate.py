# This script is used to verify that the
# implementation of `predict.py` is correct.
# To do this, we make predictions on the training
# data and compare them to the given output. 
# The resulting RMSE should be 0, since there is no noise. 
import pandas
from sklearn.metrics import mean_squared_error

train_file = 'train.csv'
result_file = 'train_results.csv'

colnames = ['Id', 'y']
csv_train = pandas.read_csv(train_file, usecols=colnames)
csv_results = pandas.read_csv(result_file, usecols=colnames)

train_ids = csv_train.Id.tolist()
result_ids = csv_results.Id.tolist()

if (len(train_ids) == len(result_ids)):
    print('Comparing y columns from ' + train_file + ' and ' + result_file + '...')
    y = csv_train.y.tolist()
    y.pop(0)                # column name
    y = [float(i) for i in y]
    y_pred = csv_results.y.tolist()
    y_pred.pop(0)           # column name
    y_pred = [float(i) for i in y_pred]
    RMSE = mean_squared_error(y, y_pred) ** 0.5
    print('Root Mean Square Error: ' + str(RMSE))
else:
    print('Ids do not match: ' + str([e for e in train_ids if e not in result_ids]))
