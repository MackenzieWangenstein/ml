import numpy as np

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
		returns:
			activation_matrix: holds an activation_vectors for each data example
			prediction_matrix: holds the real predictions for each class
		activations_matrix shape: [n x 10] matrix where x is the number of data examples
	"""
	_output_matrix = np.dot(data_inputs_matrix, weights)
	return np.where(_output_matrix > 0, 1, 0), _output_matrix
