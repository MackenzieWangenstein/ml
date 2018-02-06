import numpy as np
import perceptronutility as putil


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
				 epochs):

		self.input_node_count = training_data.shape[1]  # should be 785
		self.hidden_node_count = hidden_node_count  # 20
		self.output_node_count = data_class_count  # 10

		self.learning_rate = learning_rate
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

			# #TODO: put in a helper function?
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
		#TODO: add logic to shuffle inputs
		# for each
		output_prev_delta = 0
		hidden_prev_delta = 0

		print("original output layer weights:",  self.output_layer_weights)
		for i in range(2):  # for each training example  #self.training_data[0]) --todo: replace
			_hidden_layer_activations, _output_layer_activations = self.forward_propogate(self.training_data[i])
			_target_activations = self.training_labels[i]
			print("actual output activations: ", _output_layer_activations)
			print("target output activations: ", _target_activations)
			error = putil.sum_squared_error(_target_activations, _output_layer_activations)
			print("error for iteration ", i, ": ", error)
			#Calculate error terms for ech output unit k - slides Lecture 6 pg 37
			deltao_values = _output_layer_activations*(1-_output_layer_activations)*(_target_activations - _output_layer_activations)
			print("deltao: ", deltao_values)

			#calculate the error terms for each hidden unit j - slides Lecture 6 pg 37
			#dj = hj(1-hj) * (summation of output layer weights kj * deltao value for k)
			#^ can find values for all dj(s) at once.
			print("shape output weights: ", self.output_layer_weights.shape)

			#backprop
			delta_h_inner = np.dot( deltao_values, self.output_layer_weights.T)
			delta_h =_hidden_layer_activations * (1- _hidden_layer_activations) * delta_h_inner
			print("delta h  values : ", delta_h)


			print("shape delta o(s) shape: ", deltao_values.shape)
			print("hidden layer activations shape: ", _hidden_layer_activations)
			#update output layer weights first
			#Dw_(kj) = h*(d_k)*(h_j) where d_k = the kth deltao value & hj = activation for hidden node j

			Dwkj = self.learning_rate * np.dot(deltao_values, _hidden_layer_activations)
			print("Dwkj: ", Dwkj)
			self.output_layer_weights = self.output_layer_weights + Dwkj
			print("updated weights\n", self.output_layer_weights)
			#randomize training data order --

	# calculate error terms at each output unit
		# update weightgs

	# def output_errors(self, actual_output):
	def forward_propogate(self, data_example):
		# [1 example, 785 input values] * [785input values, 20 hidden nodes]  = [1 x 20activations]
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

		return hidden_layer_sigmoids, output_layer_sigmoids  # TODO: sigmoid activations right?
