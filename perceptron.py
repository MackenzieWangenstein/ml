import numpy as np


class Perceptron(object):
	def __init__(self,
				 learning_rate,
				 data_class_count,
				 training_matrix,
				 training_labels_matrix,
				 test_matrix,
				 test_labels_matrix,
				 initial_weights):
		"""
			Args:
				data_class_count: Number of data classes
			 	training_matrix: shape = 60,000 x (784 +1)  where + 1 is the appended input value for the bias
				training_labels_matrix: shape = 60,000 x 10
				test_matrix: shape = 10,000 x (784 +1)  where + 1 is the appended input value for the bias
				test_labels_matrix: shape = 10,000 x 10
				initial_weights: used to compare performance against perceptrons with different learning rates.
								 shape 785 x 10
		"""
		self.learning_rate = learning_rate
		self.data_class_count = data_class_count
		self.input_count = initial_weights.shape[0]
		_training_bias_col = np.ones((training_matrix.shape[0], 1))
		self.training_data = np.append(training_matrix, _training_bias_col, axis=1)
		self.training_labels = training_labels_matrix
		_test_bias_col = np.ones((test_matrix.shape[0], 1))
		self.test_data = np.append(test_matrix, _test_bias_col, axis=1)
		self.test_labels = test_labels_matrix
		self.weights = initial_weights
		self.bias = 1

		if self.weights.shape[0] != self.training_data.shape[1]:
			print("weight rows: ", self.weights.shape[0])
			print("training_data cols: ", self.training_data.shape)
			raise Exception("The numbers of rows in the weights matrix does not match the number of columns " +
							"in the training data matrix. Ensure that weights matrix contains a weight value for bias")

		if self.training_data.shape[1] != self.test_data.shape[1]:
			print("training cols: ", self.training_data.shape[1])
			print("test col cols: ", self.test_data.shape[1])
			raise Exception("The number of columns in the training data matrix does not match the number of columns " +
							"in the test data matrix")

	def prediction(self, index):
		"""
			Args:
				index(int): determines the row corresponding to the training element being examined in the training_data
							matrix


			returns the activation vector and the predicted class(in 10d vector form) for the training data
		"""

		_output_vector = np.dot(np.reshape(self.training_data[index], (self.input_count, 1)).T, self.weights)
		return np.where(_output_vector > 0, 1, 0), np.argmax(_output_vector)

	def update_weights(self, activation, target, training_data_index):
		"""
			new_weight = old_weight - learning_param*(actual - target) * input_value
			Wij = Wij - n * (Yj - Tj) * Xi

			weights shape :  785 rows(input nodes) x 10 rows(# of output nodes); Wij yields weight value
			training_date shape: 1 row x 748 columns
			activation, target shapes = [1x10]

			updates all weights for all output nodes at the same time
			requires us to change shape of training data from [1, 785] to [785, 1]

			[785, 1] * [1, x 10] gives a matrix of [785, 10]
		"""

		print("weights: ")
		dot_weights = self.learning_rate * np.dot(
			np.reshape(self.training_data[training_data_index], (self.input_count, 1)),
			(activation - target))
		print("dot weights\n", dot_weights)

		self.weights -= dot_weights;


		# self.weights = self.weights - self.learning_rate * np.dot(np.transpose([self.training_data[training_data_index]]),


def train_for_cycle(self, training_data_index):
	_activation, _actual = self.prediction(training_data_index)
	_target = self.training_labels[training_data_index]
	if _target[_actual] != 1:
		self.update_weights(_activation, _target, training_data_index)
