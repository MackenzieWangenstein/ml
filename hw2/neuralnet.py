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
		self.hidden_node_count = hidden_node_count #20
		self.output_node_count = data_class_count #10

		self.learning_rate = learning_rate
		self.hidden_layer_weights = np.random.uniform(low=-0.5, high=0.5,
									  size=(training_data.shape[1], hidden_node_count))#input nodes to hidden layer nodes
		print("shape of hidden layer weights: ", self.hidden_layer_weights.shape) # shape - 785 x 20 TODO: remove
		#columns represent the weights for node k
		self.output_layer_weights = np.random.uniform(low=-0.5, high=0.5,
									  size=(hidden_node_count, data_class_count)) #shape 20 x 10 `

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
		#for each
		for i in range(self.training_data[0]): #for each training example
			self.forward_propogate() #hidden_activations, output_activations
			#calculate error terms at each output unit
			#update weightgs


	# def output_errors(self, actual_output):
	def forward_propogate(self):
		#[60k examples, 785 input values] * [785input values, 20 hidden nodes]  = 60k x 20 activations] each column
		#represents the activation value for node k
		hidden_layer_activations = putil.predict_all(self.training_data, self.hidden_layer_weights) #shape:[785 x20]
		# TODO: save as self? ^
		# print("hidden layer activiations\n", hidden_layer_activations)  todo: remove
		# print("hidden layer activations shape: ", hidden_layer_activations.shape) todo: remove

		#use sigmoid to squash activation value for each node k for every data example   = [60k x 20]
		hidden_layer_sigmoids = putil.sigmoid_activation_values(hidden_layer_activations) #todo refactor to take just the sig
		# print("hidden layer sigmoids\n", hidden_layer_sigmoids)
		# print("hidden layer sigmoids shape", hidden_layer_sigmoids.shape)

		#[60k x 20hidden nodes] * [20 hidden nodes x 10 output nodes] = [60k x 10 ] each col is activation for node k
		output_layer_activations = putil.predict_all(hidden_layer_sigmoids, self.output_layer_weights)
		# TODO: save as self? ^
		print("output activations shape ", output_layer_activations.shape)
		output_layer_sigmoids = putil.sigmoid_activation_values(output_layer_activations)
		# print("output layer sigmoids: ", output_layer_sigmoids)
		print("output sigmoids shape", output_layer_sigmoids.shape)

		#determine error for each output unit -
		return hidden_layer_sigmoids, output_layer_sigmoids  #TODO: sigmoid activations right?


