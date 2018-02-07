import pandas as pd
import numpy as np
from hw2 import neuralnetworkrunner

print("loading training data")
# Import training data
_training_dataset_combined = pd.read_csv("mnist_train.csv")  # dataset.values returns an np array

# create training data set
_training_data = _training_dataset_combined.iloc[:, 1:785].values * (
	1 / 255)  # scale to make each value between 0 and 1
_training_labels = _training_dataset_combined.iloc[:, 0].values  # target class values are found in the first column
# add bias to input values for training data
_training_data_size = _training_dataset_combined.shape[0]
_training_bias_col = np.ones((_training_data_size, 1))
_training_data = np.append(_training_data, _training_bias_col, axis=1)
# create matrix to convert (int)class values into 10d vector values -- matrix size: trainingElementCount x class count

print("loading test data")

# create test data sets
_test_dataset = pd.read_csv("mnist_test.csv")
_test_data = _test_dataset.iloc[:, 1:785].values * (1 / 255) #TODO: investigate'
_test_labels = _test_dataset.iloc[:, 0].values
_test_data_size = _test_dataset.shape[0]
_test_bias_col = np.ones((_test_data_size, 1))
_test_data = np.append(_test_data, _test_bias_col, axis=1)
data_class_count = 10 #10 classes numbers 0-9

#
# # # ##TODO: add logic to allow developer to pass command line option to choose which to run
# perceptronrunner.run_perceptron(_training_data, _training_data_size, _training_labels, _test_data,
# 								_test_data_size, _test_labels, _rand_weights)

neuralnetworkrunner.run_neural_network(_training_data, _training_data_size, _training_labels, _test_data,
									   _test_data_size, _test_labels, data_class_count)
