import csv

# Use the following settings to generate 
# estimates for the training data, which 
# can then be used to evaluate the accuracy 
# using `evaluate.py`

#test_file = 'train.csv'
#result_file = 'train_results.csv' 

# Use the following settings to generate
# estimates for the testing data (this is the wanted result)
test_file = 'test.csv'
result_file = 'results.csv'


with open(test_file, 'r') as csv_test:
    with open(result_file, 'w+') as csv_results:
        csv_reader = csv.DictReader(csv_test)
        csv_writer = csv.DictWriter(csv_results, fieldnames = ['Id', 'y'])
        
        csv_writer.writeheader()
        for row in csv_reader:
            id = row['Id']

            # Compute mean
            mean = 0
            n = 0
            for i in range(1, 11):
                x_i = float(row['x' + str(i)])
                mean = mean + x_i
                n = n + 1
            mean = mean / n

            # Write to file
            csv_writer.writerow({'Id': id, 'y': mean})
