# This script is used to verify that the
# implementation of `predict.py` is correct.
# To do this, we make predictions on the training
# data and compare them to the given output. 
# The resulting RMSE should be 0, since there is no noise. 
import csv


train_file = 'train.csv'
result_file = 'train_results.csv'

with open(train_file, 'r') as csv_train:
    with open(result_file, 'r') as csv_results:
        train_reader = csv.DictReader(csv_train)
        res_reader = csv.DictReader(csv_results)
        print('Comparing y columns from ' + train_file + ' and ' + result_file + '...')
        n = 0
        RMSE = 0
        for train_row in train_reader:
            for res_row in res_reader:
                if train_row['Id'] == res_row['Id']:
                    y = float(train_row['y'])
                    y_pred = float(res_row['y'])
                    RMSE = RMSE + (y - y_pred) ** 2.0
                    n = n + 1
        RMSE = (RMSE / n) ** 0.5
        print('Root Mean Square Error: ' + str(RMSE))
