import pandas as pd
import numpy as np

from perceptron import Perceptron

data_class_count = 10;
epochs = 2 # TODO: replace with 70
n1 = 0.001
n2 = 0.01
n3 = 0.1

# Import training data
_training_dataset_combined = pd.read_csv("mnist_train.csv")  # dataset.values returns an np array
_training_data = _training_dataset_combined.iloc[:, 1:783].values * (
	1 / 255)  # scale to make each value between 0 and 1

# create training data set
_training_labels = _training_dataset_combined.iloc[:, 0].values  # target class values are found in the first column
_training_data_size = _training_dataset_combined.shape[0]
# create matrix to convert (int)class values into 10d vector values -- matrix size: trainingElementCount x class count
_training_labels_matrix = np.zeros((_training_data_size, 10))
for i in range(_training_data_size):
	_train_target_output = _training_labels[i]
	_training_labels_matrix[i][_train_target_output] = 1

# create test data sets
_test_dataset = pd.read_csv("mnist_test.csv")
_test_data = _test_dataset.iloc[:, 1:783].values * (1 / 255)
_test_labels = _test_dataset.iloc[:, 0].values
_test_data_size = _test_dataset.shape[0]
_test_labels_matrix = np.zeros((_test_data_size, data_class_count))
for j in range(_test_data_size):
	_test_target_output = _test_labels[j]
	_test_labels_matrix[j][_test_target_output] = 1

# create weights matrix for training set 10 rows with 748 randomly generated weights from -0.5to 0.5
# Wij represents weight connecting input node i to output onde j
_rand_weights = np.random.uniform(low=-0.5, high=0.5,
								  size=(_training_data.shape[1] + 1, 10))  # include weight for bias

p1 = Perceptron(n1, data_class_count, _training_data, _training_data_size, _training_labels_matrix, _test_data,
				_test_data_size, _test_labels_matrix, _rand_weights, epochs)
p2 = Perceptron(n2, data_class_count, _training_data, _training_data_size, _training_labels_matrix, _test_data,
				_test_data_size, _test_labels_matrix, _rand_weights, epochs)
p3 = Perceptron(n3, data_class_count, _training_data, _training_data_size, _training_labels_matrix, _test_data,
				_test_data_size, _test_labels_matrix, _rand_weights, epochs)

p1_epochs_ran, p1_training_accuracy, p1_test_accuracy = p1.run()
print("Perceptron of learning rate ", n1, " had a final training accuracy of ", p1_training_accuracy, " and a test",
	  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")

p2_epochs_ran, p2_training_accuracy, p2_test_accuracy = p2.run()
print("Perceptron of learning rate ", n2, " had a final training accuracy of ", p2_training_accuracy, " and a test",
	  " accuracy of ", p1_test_accuracy, " after ", p2_epochs_ran, " epochs")

p3_epochs_ran, p3_training_accuracy, p3_test_accuracy = p3.run()
print("Perceptron of learning rate ", n3, " had a final training accuracy of ", p3_training_accuracy, " and a test",
	  " accuracy of ", p1_test_accuracy, " after ", p3_epochs_ran, " epochs")
