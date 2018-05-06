import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import keras.optimizers

from keras import backend as K

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

n_train = train.index.size
cols = train.values.shape[1]
y_train = train.values[:, 0]
X_train = train.values[:, 1:cols].astype('float32')
input_dim = X_train.shape[1]
num_classes = 5

n_test = test.index.size
X_test = test.values.astype('float32')

assert(X_test.shape[1] == X_train.shape[1])

y_train = keras.utils.to_categorical(y_train, num_classes)

# TODO: Preprocess data to zero mean, unit variance?

# Train several NN to avoid being stuck at local optimum
num_runs = 3
best_score = 0
best_NN = None
for run in range(num_runs):
    print("Training NN number " + str(run))
    # Build NN
    act = 'relu'    # Activation function
    dropP = 0.3     # Dropout probability
    num_layers = 1 # Number of hidden layers
    num_nodes = 200 # Number of nodes at each layer

    NN = Sequential()
    NN.name = 'NN' + str(run)

    # Input layer
    NN.add(Dense(num_nodes, activation = act, input_dim = input_dim))

    # Hidden layers
    for l in range(num_layers):
        NN.add(Dense(num_nodes, activation = act))
        NN.add(Dropout(dropP))

    # Output layers
    NN.add(Dense(num_classes, activation = 'softmax'))

    # Compile NN
    optimizer = keras.optimizers.Adadelta()
    NN.compile( loss = 'categorical_crossentropy', 
                optimizer = optimizer, 
                metrics = ['accuracy'])

    # Train NN
    batch_size = 32 # Number of batches used for SGD
    epochs = 1     # Number of passes through data
    history = NN.fit(X_train, y_train, 
                        batch_size = batch_size, 
                        epochs = epochs,
                        verbose = 1,
                        validation_split = 0.4)

    # Set best NN
    val_acc = history.history['val_acc'][-1] # Last accuracy score on validation set
    if val_acc > best_score:
        best_NN = NN

print("Best validation accuracy score: " + str(best_score))

# Predict class labels
y_test = best_NN.predict(X_test)

# Convert from categorical to class label
y_test = np.argmax(y_test, axis = 1)
print(y_test)

# Write results to file
res_frame = pd.DataFrame(y_test, index = test.index)
res_frame.to_hdf("res.h5", key = 'res')















