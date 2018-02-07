import numpy as np
import perceptronutility as putil


class NeuralNet(object):
	def __init__(self,
				 hidden_node_count,
				 learning_rate,
	             momentum,
				 data_class_count,
				 training_data,
				 training_count,
				 training_labels_matrix,
				 test_data,
				 test_data_size,
				 test_labels_matrix,
				 epochs):

		"""

		:param hidden_node_count:
		:param learning_rate:
		:param momentum:
		:param data_class_count:
		:param training_data: the inputs associated with each training example
		:param training_count:
		:param training_labels_matrix: the expected class value in vector form for each training example
		:param test_data:   the inputs associated with each test example
		:param test_data_size:
		:param test_labels_matrix: the expected class value in vector form for each test example
		:param epochs:

		Notes:
					Confusion_matrix: The confusion_matrix[i][j] gives us the number of data_examples that were placed
					into that classification case. i is the actual predicted class for the training element, where j is
					the target class value. Diagonal values represents true positive predictions, while all other values
					are considered false positive predictions.

					accuracy = sum of all correct predictions(confusion_matrix diagonal values) / sum of all predictions
		"""

		self.input_node_count = training_data.shape[1]  # should be 785
		self.hidden_node_count = hidden_node_count  # 20
		self.output_node_count = data_class_count  # 10

		self.learning_rate = learning_rate
		self.momentum = momentum
		self.hidden_layer_weights = np.random.uniform(low=-0.5, high=0.5,
													  size=(training_data.shape[1],
															hidden_node_count))  # input nodes to hidden layer nodes
		print("shape of hidden layer weights: ", self.hidden_layer_weights.shape)  # shape - 785 x 20 TODO: remove
		# columns represent the weights for node k
		self.output_layer_weights = np.random.uniform(low=-0.5, high=0.5,
													  size=(hidden_node_count + 1, data_class_count))  # shape 21 x 10 `

		self.training_data = training_data
		self.training_data_size = training_count
		self.training_labels = training_labels_matrix
		self.test_data = test_data
		self.test_data_size = test_data_size
		self.test_labels = test_labels_matrix

		self.bias = 1
		self.epochs = epochs
		self.training_confusion_matrix = np.zeros((self.output_node_count, self.output_node_count))
		self.test_confusion_matrix = np.zeros((self.output_node_count, self.output_node_count))

		if self.hidden_layer_weights.shape[0] != self.training_data.shape[1]:
			print("weight rows: ", self.hidden_layer_weights.shape[0])
			print("training_data cols: ", self.training_data.shape)
			raise Exception("The numbers of rows in the weights matrix does not match the number of columns " +
							"in the training data matrix. Ensure that weights matrix contains a weight value for bias")

		if self.training_data.shape[1] != self.test_data.shape[1]:
			print("training cols: ", self.training_data.shape[1])
			print("test col cols: ", self.test_data.shape[1])
			raise Exception("The number of columns in the training data matrix does not match the number of columns " +
							"in the test data matrix")

	def run(self):
		for i in range(self.epochs):
			self.training_cycle()  # get confusion matrix
			print("after training cycle")
			# # get activations(as sigmoids) for data examples with final weights --
			training_hidden_layer_activations, training_output_layer_activations = self.forward_propgate_all(self.training_data)
			print("hidden layer activations for training: ", training_hidden_layer_activations)
			print("output layer activations for training: ", training_output_layer_activations)

			# _training_actual = np.argmax(training_output_layer_activations)
			# _training_target = np.where(self.training_labels[i] == 0.9)[0]
			# print("training actual ", _training_actual)
			# print("training target ", _training_target)
			# self.training_confusion_matrix[_training_target, _training_actual] += 1
			#
			# _curr_training_accuracy = putil.compute_accuracy(self.training_confusion_matrix)
			# _test_accuracy = putil.compute_accuracy(self.test_confusion_matrix)
			#
			# test_hidden_layer_activations, test_output_layer_activations = self.forward_propgate_all(self.training_data)
			# print("")

			# # predict on training set
			# training_predictions = putil.predict_all(self.training_data, self.weights)
			# for element_index in range(self.training_data_size):
			# 	_training_actual = np.argmax(training_predictions[element_index])
			# 	_training_target = np.where(self.training_labels[element_index] == 1)[0]  # expected class as int
			# 	self.training_confusion_matrix[_training_target, _training_actual] += 1
			#
			# # predict on test set
			# test_predictions = putil.predict_all(self.test_data, self.weights)
			# for element_index in range(self.test_data_size):
			# 	_test_actual = np.argmax(test_predictions[element_index])
			# 	_test_target = np.where(self.test_labels[element_index] == 1)[0]
			# 	self.test_confusion_matrix[_test_target, _test_actual] += 1
			#
			# _curr_training_accuracy = putil.compute_accuracy(self.training_confusion_matrix)
			# _test_accuracy = putil.compute_accuracy(self.test_confusion_matrix)
			# print("training accuracy: ", _curr_training_accuracy)
			# print("test acurracy: ", _test_accuracy)

	def training_cycle(self):

		output_prev_delta = np.zeros(np.shape(	self.output_layer_weights))
		hidden_prev_delta = np.zeros(np.shape(self.hidden_layer_weights))

		#TODO: add logic to shuffle inputs

		print("original output layer weights:",  self.output_layer_weights)
		for i in range(2):  # for each training example  #self.training_data[0]) --todo: replace
			print("i = ", i)
			#np.reshape(self.training_data[i], (1, len(self.training_data[i])))
			_hidden_layer_activations, _output_layer_activations = self.forward_propogate(self.training_data[i])
			_target_activations = self.training_labels[i]
			# print("actual output activations: ", _output_layer_activations) TODO: remove
			# print("target output activations: ", _target_activations)
			error = putil.sum_squared_error(_target_activations, _output_layer_activations)
			# print("error for iteration ", i, ": ", error) TODO: remove
			#Calculate error terms for ech output unit k - slides Lecture 6 pg 37
			delta_o_values = _output_layer_activations*(1-_output_layer_activations)*(_target_activations - _output_layer_activations)
			# print("deltao: ", delta_o_values) TODO: remove

			#calculate the error terms for each hidden unit j - slides Lecture 6 pg 37
			#dj = hj(1-hj) * (summation of output layer weights kj * deltao value for k)
			#^ can find values for all dj(s) at once.
			delta_h_inner = np.dot(delta_o_values, self.output_layer_weights.T)
			delta_h_values =_hidden_layer_activations * (1- _hidden_layer_activations) * delta_h_inner
			# print("delta h  values : ", delta_h_values) TODO: remove

			# print("shape delta o(s) shape: ", delta_o_values.shape)
			# print("hidden layer activations shape: ", _hidden_layer_activations)

			#updated output layer weights = Dwkj + momentum * previous weight change
			#Dwkj = h*(d_k)*(h_j) where h is the learning rate, d_k = the kth deltao value & hj = activation for hidden node j
			Dwkj = self.learning_rate * np.dot( _hidden_layer_activations.T, delta_o_values) + self.momentum * output_prev_delta
			# print("Dwkj: ", Dwkj) TODO: remove
			#output layer weights = 21 x 10

			self.output_layer_weights = self.output_layer_weights + Dwkj
			output_prev_delta = self.output_layer_weights

			data_example = np.reshape(self.training_data[i], (1, self.training_data[i].shape[0]))

			print("data example shape", data_example.shape)
			print("delta_h_values shape, ", delta_h_values.shape)

			#slides 45  - shape of hidden weights = [785 x 21]

																				#strip off bias
			Dwji = self.learning_rate * np.dot(data_example.T, delta_h_values[:,:-1]) + self.momentum * hidden_prev_delta
			self.hidden_layer_weights = self.hidden_layer_weights + Dwji
			hidden_prev_delta = self.hidden_layer_weights
			print("updated weights\n", self.output_layer_weights)
			#randomize training data order needed for sequential data according to slide 40
			# self.training_data = np.random.shuffle(self.training_data) #TODO: bug here - not scriptable



	def forward_propgate_all(self, data_examples_set):
		"""

		:param data_examples_set:
		:return: hidden_layer_sigmoids shape: [n x 21]
				 output_layer_sigmoids shape: [n x 10]
				where n is number of training examples in data_set
		"""
		#shape [n  x 20 +1] activations where n = number of data examples in set and 1 bias node is appended
		_input_bias_col = np.ones((np.shape(data_examples_set)[0], 1))
		# print("bias_col shape: ", _input_bias_col) #training example already has bias
		# data_examples_set = np.append(data_examples_set, _input_bias_col, axis=1)
		hidden_layer_activations = np.dot(data_examples_set, self.hidden_layer_weights)
		hidden_layer_sigmoids = putil.sigmoid_activation_values_all(hidden_layer_activations)

		_hidden_bias_col = np.ones((np.shape(hidden_layer_sigmoids)[0], 1))
		hidden_layer_sigmoids = np.append(hidden_layer_sigmoids, _hidden_bias_col, axis=1)

		output_layer_activations = np.dot(hidden_layer_sigmoids, self.output_layer_weights)
		output_layer_sigmoids = putil.sigmoid_activation_values_all(output_layer_activations)
		return hidden_layer_sigmoids, output_layer_sigmoids


	# def output_errors(self, actual_output):
	def forward_propogate(self, data_example):
		# [1 example x 785 input values] * [785input values x 20 hidden nodes]  = [1 x 20activations]
		# each column represents the activation value for node k
		data_example = np.reshape(data_example, (1, len(data_example)))

		hidden_layer_activations = np.dot(data_example, self.hidden_layer_weights)  # shape: [785 x20]

		# use sigmoid to squash activation value for each node k for every data example   = [60k x 20]
		hidden_layer_sigmoids = putil.sigmoid_activation(hidden_layer_activations)

		# add bias  so that inputs into output layer is 20 hidden layer activations + (1 bias)
		hidden_layer_sigmoids = np.concatenate((hidden_layer_sigmoids,
												np.ones((np.shape(hidden_layer_sigmoids)[0],1))),
											   axis=1)

		# [1 x (20hidden nodes + 1bias)] * [(20 hidden nodes + 1bias) x 10 output nodes] = [1 x 10 ]
		# each col is activation for output node k

		output_layer_activations = np.dot(hidden_layer_sigmoids, self.output_layer_weights)
		output_layer_sigmoids = putil.sigmoid_activation(output_layer_activations)
		print("output sigs")
		print(output_layer_sigmoids)
		print("hidden sigmods")
		print(hidden_layer_sigmoids)
		return hidden_layer_sigmoids, output_layer_sigmoids

