import numpy as np
import time
import perceptron_utility as putil


class Perceptron(object):
	# create file that stores the results of training predictions between epochs
	directory = './save/{}'.format(time.ctime().replace(' ', '-'))
	file = directory + '/{}'

	def __init__(self,
				 learning_rate,
				 data_class_count,
				 training_data,
				 training_count,
				 training_labels_matrix,
				 test_data,
				 test_data_size,
				 test_labels_matrix,
				 initial_weights,
				 epochs):
		"""
			Args:
				data_class_count: Number of data classes
			 	training_matrix: shape = 60,000 x (784 +1)  where + 1 is the appended input value for the bias
				training_labels_matrix: shape = 60,000 x 10
				test_data: shape = 10,000 x (784 +1)  where + 1 is the appended input value for the bias
				test_labels_matrix: shape = 10,000 x 10
				initial_weights: used to compare performance against perceptrons with different learning rates.
								 shape 785 x 10

			Notes:
					Confusion_matrix: The confusion_matrix[i][j] gives us the number of data_examples that were placed
					into that classification case. i is the actual predicted class for the training element, where j is
					the target class value. Diagonal values represents true positive predictions, while all other values
					are considered false positive predictions.

					accuracy = sum of all correct predictions(confusion_matrix diagonal values) / sum of all predictions
		"""
		self.learning_rate = learning_rate
		self.data_class_count = data_class_count
		self.input_node_count = initial_weights.shape[0]  # should be 785
		_training_bias_col = np.ones((training_data.shape[0], 1))
		self.training_data = np.append(training_data, _training_bias_col, axis=1)
		self.training_data_size = training_count
		self.training_labels = training_labels_matrix
		_test_bias_col = np.ones((test_data.shape[0], 1))
		self.test_data = np.append(test_data, _test_bias_col, axis=1)
		self.test_data_size = test_data_size
		self.test_labels = test_labels_matrix
		self.weights = initial_weights
		self.bias = 1
		self.epochs = epochs
		self.training_confusion_matrix = np.zeros((self.data_class_count, self.data_class_count))
		self.test_confusion_matrix = np.zeros((self.data_class_count, self.data_class_count))

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
	# TODO: remove
	# def predict_all(self, data_inputs_matrix):
	# 	"""
	# 		returns:
	# 			activation_matrix: holds an activation_vectors for each data example
	# 			prediction_matrix: holds the real predictions for each class
	# 		activations_matrix shape: [n x 10] matrix where x is the number of data examples
	# 	"""
	# 	_output_matrix = np.dot(data_inputs_matrix, self.weights)
	#
	# 	return np.where(_output_matrix > 0, 1, 0), _output_matrix

	def prediction(self, data_inputs_arr):
		"""
			Args:
				index(int): determines the row corresponding to the training element being examined in the training_data
							matrix
			returns the activation vector and the predicted class(in 10d vector form) for the training data

		"""
		data_values = np.reshape(data_inputs_arr, (self.input_node_count, 1))  # converts to [n, 1] matrix - unnecessary
		_output_vector = np.dot(data_values.T, self.weights)
		return np.where(_output_vector > 0, 1, 0), np.argmax(_output_vector)

	#//TODO: keep or remove
	# def update_weights(self, activation, target, training_data_element):
	# 	"""
	# 		new_weight = old_weight - learning_param*(actual - target) * input_value
	# 		Wij = Wij - n * (Yj - Tj) * Xi
	#
	# 		weights shape :  785 rows(input nodes) x 10 rows(# of output nodes); Wij yields weight value
	# 		training_date shape: 1 row x 748 columns
	# 		activation, target shapes = [1x10]
	#
	# 		updates all weights for all output nodes at the same time
	# 		requires us to change shape of training data from [1, 785] to [785, 1]
	#
	# 		[785, 1] * [1, x 10] gives a matrix of [785, 10]
	#
	# 		reshaping training_Data_element acts like a tranpspose
	# 	"""
	# 	training_data_element = np.reshape(training_data_element, (self.input_node_count, 1))
	# 	self.weights -= self.learning_rate * np.dot(training_data_element, (activation - target))

	def run(self):
		"""
			train hw1 for n cycles or until accuracy difference between cycles is less than 1%
			where n = number of epochs
		:return: number of training cycles the hw1 trained for, return accuracies for test and training sets
		"""
		self.actual_training_cycles = 0
		prev_training_accuracy = 0;

		print("Training hw1 with learning rate of ", self.learning_rate)
		for i in range(self.epochs):
			self.train_for_cycle()

			# predict on training set
			training_input_activations, training_predictions = putil.predict_all(self.training_data, self.weights)
			for element_index in range(self.training_data_size):
				_training_actual = np.argmax(training_predictions[element_index])
				_training_target = np.where(self.training_labels[element_index] == 1)[0]  # expected class as int
				self.training_confusion_matrix[_training_target, _training_actual] += 1

			# predict on test set
			test_input_activations, test_predictions = putil.predict_all(self.test_data, self.weights)
			for element_index in range(self.test_data_size):
				_test_actual = np.argmax(test_predictions[element_index])
				_test_target = np.where(self.test_labels[element_index] == 1)[0]
				self.test_confusion_matrix[_test_target, _test_actual] += 1

			_curr_training_accuracy = putil.compute_accuracy(self.training_confusion_matrix)
			print("training acc for ", self.learning_rate, " epoch ", i, ": ", _curr_training_accuracy)

			_test_accuracy = putil.compute_accuracy(self.test_confusion_matrix)
			print("test acc for ", self.learning_rate, " epoch ", i, ": ", _test_accuracy)

			# return if training accuracy does not improve by more than 1% between epochs
			if _curr_training_accuracy - prev_training_accuracy < 0.01:
				return i, _curr_training_accuracy, _test_accuracy
			prev_training_accuracy = _curr_training_accuracy

		return self.epochs, _curr_training_accuracy, _test_accuracy

	def train_for_cycle(self):
		for i in range(0, self.training_data.shape[0]):
			_training_data_element = self.training_data[i]
			_activation, _actual = self.prediction(self.training_data[i])  # replace with data
			_target = self.training_labels[i]
			if _target[_actual] != 1:
				self.weights = putil.update_weights(_activation,
													_target,
													_training_data_element,
													self.weights,
													self.learning_rate,
													self.input_node_count)
			# self.update_weights(_activation, _target, training_data_element) TODO: keep or remove

