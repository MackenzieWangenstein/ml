from hw2.neuralnet import NeuralNet

import numpy as np
import math
import sys


# TODO: remove args to supply training data size and test data size -- use shape instead
def run_neural_network(training_data,
					   training_labels,
					   test_data,
					   test_labels,
					   data_class_count,
					   learning_rate):
	training_data_size = training_data.shape[0]
	test_data_size = test_data.shape[0]
	momentum_default = 0.9
	momentum_zero = 0
	momentum_quartile = 0.25
	momentum_half = 0.50
	epochs = 50  # 50  # TODO: 50
	hidden_nodes_twenty = 20
	hidden_nodes_fifty = 50
	hidden_nodes_hundred = 100
	training_labels_matrix = np.full((training_data_size, data_class_count), 0.1)
	run_all = False
	if sys.argv.__contains__("-all"):
		run_all = True

	print("run all ", run_all)

	for i in range(training_data_size):
		_train_target_output = training_labels[i]
		training_labels_matrix[i][_train_target_output] = 0.9

	test_labels_matrix = np.full((test_data_size, data_class_count), 0.1)
	for j in range(test_data_size):
		_test_target_output = test_labels[j]
		test_labels_matrix[j][_test_target_output] = 0.9

	"""Experiment 1"""
	"""nn1 - hidden node count = 20, momentum = 0.9"""
	if run_all or sys.argv.__contains__("-nn1"):
		run_experiment(hidden_nodes_twenty, learning_rate, momentum_default, data_class_count, training_data,
					   training_labels_matrix, test_data, test_labels_matrix, epochs, experiment_name="nn1")

	"""nn2 - hidden node count = 50, momentum = 0.9"""
	if run_all or sys.argv.__contains__("-nn2"):
		run_experiment(hidden_nodes_fifty, learning_rate, momentum_default, data_class_count, training_data,
					   training_labels_matrix, test_data, test_labels_matrix, epochs, "nn2")

	"""nn3 - hidden node count = 100, momentum = 0.9"""
	if run_all or sys.argv.__contains__("-nn3"):
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_default, data_class_count, training_data,
					   training_labels_matrix, test_data, test_labels_matrix, epochs, experiment_name="nn3")

	"""Experiment 2: Vary momentum"""
	"""nn4 - momentum = 0, hidden node count = 100 """
	if run_all or sys.argv.__contains__("-nn4"):
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_zero, data_class_count, training_data,
					   training_labels_matrix, test_data, test_labels_matrix, epochs, experiment_name="nn4")

	"""nn experiment 2: momentum = 0.25, hidden node count = 100 """
	if run_all or sys.argv.__contains__("-nn5"):
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_quartile, data_class_count, training_data,
					   training_labels_matrix, test_data, test_labels_matrix, epochs, experiment_name="nn5")

	"""nn experiment 2: momentum = 0.5, hidden node count = 100 """
	if run_all or sys.argv.__contains__("-nn6"):
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_half, data_class_count, training_data,
					training_labels_matrix, test_data, test_labels_matrix, epochs, experiment_name="nn6")

	"""Experiment 3: vary training examples"""
	"""nn7 -  experiment 3:  training data size = 29,999, momentum = 0.9, hidden node count = 100 """
	if run_all or sys.argv.__contains__("-nn7"):
		half_data_set, half_data_labels_matrix = balanced_subset_data(training_data,
																	  training_labels,
																	  training_labels_matrix,
																	  data_class_count, 0.5)
		print("half data shape: ", half_data_set.shape)
		print("Half data labels shape: ", half_data_labels_matrix.shape)
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_default, data_class_count, half_data_set,
					   half_data_labels_matrix, test_data, test_labels_matrix, epochs, "nn7")

	"""nn8 -  experiment 3:  training data size = 29,999, momentum = 0.9, hidden node count = 100 """
	if run_all or sys.argv.__contains__("-nn8"):
		quarter_data_set, quarter_data_labels_matrix = balanced_subset_data(training_data,
																	  training_labels,
																	  training_labels_matrix,
																	  data_class_count, 0.25)
		print("quarter data shape: ", quarter_data_set.shape)
		print("quarter data labels shape: ", quarter_data_labels_matrix.shape)
		run_experiment(hidden_nodes_hundred, learning_rate, momentum_default, data_class_count, quarter_data_set,
					   quarter_data_labels_matrix, test_data, test_labels_matrix, epochs, "nn8")


# TODO: refactor remove params: data_class count and training_labels - use np.where and shape on label_matrix instead
def balanced_subset_data(training_data, training_labels, training_labels_matrix, data_class_count,
						 desired_data_percentage):
	"""
		creates a subset of data examples where data is approximately balanced
	"""
	desired_entries_count = math.floor(training_data.shape[0] * desired_data_percentage)
	items_needed_per_class = math.floor(desired_entries_count / data_class_count)
	added_examples_count = 0
	training_data = np.array(training_data)
	# np.random.shuffle(training_data) #TODO: determine if we care to have data shuffled each time
	subset = np.zeros((desired_entries_count, training_data.shape[1]))
	subset_labels_matrix = np.zeros((desired_entries_count, training_labels_matrix.shape[1]))
	print("subset_labels_matrix shape", subset_labels_matrix.shape)
	not_added_indices = []

	# used to keep track of how many data examples there are for each data class- ensures we get a balanced subset
	class_tracker_map = np.zeros(data_class_count)
	for i in range(training_data.shape[0]):
		if added_examples_count == desired_entries_count:
			break
		if not class_count_exceeded(class_tracker_map, items_needed_per_class, training_labels[i]):
			subset[added_examples_count] = training_data[i]
			subset_labels_matrix[added_examples_count] = training_labels_matrix[i]
			added_examples_count += 1
			class_tracker_map[training_labels[i]] += 1
		else:
			not_added_indices.append(i)

	if added_examples_count != desired_entries_count:
		print("Warning! could not create a subset of length ", desired_entries_count, " with balanced data.",
			  " appending data examples at random to reach subset size requirements")
		j = 0
		while added_examples_count != desired_entries_count and j < len(not_added_indices):
			index = not_added_indices[j]
			subset[added_examples_count] = training_data[index]
			subset_labels_matrix[added_examples_count] = training_labels_matrix[j]
			added_examples_count += 1
			class_tracker_map[training_labels[j]] += 1
			j += 1

	if added_examples_count != desired_entries_count:
		raise Exception("Something went wrong! ran out of examples while creating subset. bug here!")

	if np.unique(subset, axis=0).shape[0] != desired_entries_count:
		raise Exception("Duplicate training example found!")

	print("final class representation of training data subset: ", class_tracker_map)
	print("subset matrix shape before return: ", subset_labels_matrix.shape)
	return subset, subset_labels_matrix


def class_count_exceeded(class_tracker_map, items_needed_per_class, data_example_class):
	count = class_tracker_map[data_example_class]
	if count >= items_needed_per_class:
		return True
	else:
		return False


def run_experiment(hidden_nodes,
				   learning_rate,
				   momentum,
				   data_class_count,
				   training_data,
				   training_labels_matrix,
				   test_data,
				   test_labels_matrix,
				   epochs,
				   experiment_name):
	nn = NeuralNet(hidden_nodes, learning_rate, momentum, data_class_count, training_data, training_labels_matrix,
				   test_data, test_labels_matrix, epochs)

	print("training data shape", training_data.shape)  # TODO: remove
	print("training labels matrix shape: ", training_labels_matrix.shape)
	nn_epochs_ran, nn_training_accuracy, nn_test_accuracy = nn.run()  # TODO: remove nn_confu matrix
	nn.plot_accuracy_history("hw2/results/" + experiment_name + ".png")
	nn.save_final_results("hw2/results/" + experiment_name + ".txt")
	# nn.plot_error_history()
	print("Perceptron with momentum ", momentum, "and 100 hidden nodes had a final training accuracy of ",
		  nn_training_accuracy, " and a test accuracy of ", nn_test_accuracy, "after ", nn_epochs_ran, " epochs")
	nn.display_prediction_history()