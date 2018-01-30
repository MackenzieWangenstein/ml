import pandas as pd
import numpy as np

from perceptron import Perceptron

n1 = 0.001
n2 = 0.01
n3 = 0.1

data_class_count = 10;

epochs = 10  # TODO: replace with 70

print("test main")
# Import training data
_training_dataset_combined = pd.read_csv("mnist_train.csv")  # dataset.values returns an np array
_training_data = _training_dataset_combined.iloc[:, 1:783].values * (
	1 / 255)  # scale to make each value between 0 and 1

# create training data set
_training_labels = _training_dataset_combined.iloc[:, 0].values  # target class values are found in the first column
_training_count = _training_dataset_combined.shape[0]
# create matrix to convert (int)class values to 10d vector values -- matrix size: trainingElementCount x class count

_training_labels_matrix = np.zeros((_training_count, 10))
for i in range(_training_count):
	_target_ouput = _training_labels[i]
	_training_labels_matrix[i][_target_ouput] = 1

# uncomment to see that class conversion worked
# print(np.array([_training_labels]).T)  #print
# print(_training_labels_matrix)

# create test data sets
_test_dataset = pd.read_csv("mnist_test.csv")
_test_data = _test_dataset.iloc[:, 1:783].values * (1 / 255)
_test_labels = _test_dataset.iloc[:, 0].values
_test_count = _test_dataset.shape[0]
_test_labels_matrix = np.zeros((_test_count, data_class_count))
for j in range(_test_count):
	_target_output = _test_labels[j]
	_test_labels_matrix[j][_target_ouput] = 1



# #create weights matrix for training set 10 rows with 748 randomly generated weights from -0.5to 0.5
# Wij represents weight connecting input node i to output onde j
_rand_weights = np.random.uniform(low=-0.5, high=0.5,
								  size=(_training_data.shape[1] + 1,10))  # include weight for bias
# print("Initial weights: \n", _rand_weights)

p1 = Perceptron(n1, data_class_count, _training_data, _training_labels_matrix, _test_data, _test_labels_matrix, _rand_weights)
p2 = Perceptron(n2, data_class_count, _training_data, _training_labels_matrix, _test_data, _test_labels_matrix, _rand_weights)
p3 = Perceptron(n3, data_class_count, _training_data, _training_labels_matrix, _test_data, _test_labels_matrix, _rand_weights)

# initialization of weights counts as the 0 epoch
for i in range(1, epochs):
	print("prior weights: ")
	print(p3.weights)

	for j in range(0, _training_count):
		p1.train_for_cycle(j)
		p2.train_for_cycle(j)
		p3.train_for_cycle(j)

	print("updated weights: ")
	print(p3.weights)

# predict
# train
