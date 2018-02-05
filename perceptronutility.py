import numpy as np
import math


def sigmoid_activation_values(layer_activations):
	"""
		args:
				layer_activations is a [data_examples x c] matrix where c is the number of nodes in the current layer.
	return:
		returns an [data_xamples x c] array of  sigmoid activations.
		the sigmoid value(between 0 and 1) indicates if the node fires or not.
	note:
		dot products with a value a little greater than zero will give us  a sigmoid value a little greater than .5
		dot products with a value a little less than zero will giv us a sigmoid value of a little less than .5
	"""
	data_example_count = layer_activations.shape[0]  # 60k
	node_activations_count = layer_activations.shape[1]  # n
	print("activations count")
	sigmoid_activations = np.zeros((data_example_count, node_activations_count))  # 60k x n nodes
	for i in range(data_example_count):  # for every training example  //todo: is there a better way than o(n^2)
		for j in range(node_activations_count):
			sigmoid_activations[i][j] = 1 / (1 + math.exp(layer_activations[i][j]))
	return sigmoid_activations


##TODO: remove
def update_weights(activation, target, training_data_element, weights, learning_rate, input_node_count):
	"""
		new_weight = old_weight - learning_param*(actual - target) * input_value
		Wij = Wij - n * (Yj - Tj) * Xi

		weights shape :  785 rows(input nodes) x 10 rows(# of output nodes); Wij yields weight value
		training_date shape: 1 row x 748 columns
		activation, target shapes = [1x10]

		updates all weights for all output nodes at the same time
		requires us to change shape of training data from [1, 785] to [785, 1]

		[785, 1] * [1, x 10] gives a matrix of [785, 10]

		reshaping training_Data_element acts like a tranpspose
	"""
	training_data_element = np.reshape(training_data_element, (input_node_count, 1))
	weights -= learning_rate * np.dot(training_data_element, (activation - target))
	return weights


def compute_accuracy(confusion_matrix):
	"""
	computes the accuracy of the predictions for the data set associated with the confusion matrix
	accuracy = sum of diagonals in confusion matrix / sum of all elements in confusion matrix

	Notes:
	np.trace sums the diagonal values in a matrix.
	"""
	return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def predict_all(data_inputs_matrix, weights):
	"""
		args:
			data_inputs_matrix: a list of the input values for each data example in the associated datas et
			weights: weights will be the weight matrix associated with either a input_layer or a hidden_layer
				input_layer_weights shape:  [785 x n]    where n is the number of hidden nodes
				hidden_layer_weights shape for [n x 10]  where n is number of hidden nodes and 10 is # of output nodes
	return:
			activation_matrix: holds an activation_vectors for each data example
			prediction_matrix: holds the real predictions for each class
		activations_matrix shape: [n x c] matrix where x is the number of data examples and c is
	"""
	_output_matrix = np.dot(data_inputs_matrix, weights)
	return _output_matrix
