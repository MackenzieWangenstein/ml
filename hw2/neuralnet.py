import numpy as np


class NeuralNet(object):
	def __init__(self,
				 hidden_node_count,
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

		self.hidden_node_count = hidden_node_count
		self.learning_rate = learning_rate
		self.data_class_count = data_class_count
		self.input_node_count = initial_weights.shape[0]  # should be 785
		self.training_data = training_data
		self.training_data_size = training_count
		self.training_labels = training_labels_matrix
		self.test_data = test_data
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
