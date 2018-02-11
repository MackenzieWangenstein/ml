from hw2.neuralnet import NeuralNet

import numpy as np
import math


def run_neural_network(training_data,
					   training_data_size,
					   training_labels,
					   test_data,
					   test_data_size,
					   test_labels,
					   data_class_count,
					   learning_rate):
	print("data class count", data_class_count)
	# learning_rate = 0.1
	momentum_default = 0.9
	momentum_zero = 0
	momentum_quartile = 0.25
	momentum_half = 0.50
	epochs = 1
	# epochs = 50
	hidden_nodes1 = 20
	hidden_nodes2 = 50
	hidden_nodes_hundred = 100

	training_labels_matrix = np.full((training_data_size, data_class_count), 0.1)
	for i in range(training_data_size):
		_train_target_output = training_labels[i]
		training_labels_matrix[i][_train_target_output] = 0.9

	test_labels_matrix = np.full((test_data_size, data_class_count), 0.1)
	for j in range(test_data_size):
		_test_target_output = test_labels[j]
		test_labels_matrix[j][_test_target_output] = 0.9

	"""nn experiment 1  - hidden node count = 20, momentum = 0.9"""

	# nn1 = NeuralNet(hidden_nodes1, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# p1_epochs_ran, p1_training_accuracy, p1_test_accuracy, p1_confusion_mat = nn1.run()
	# nn1.plot_accuracy_history("hw2/results/nn1.png")
	# nn1.save_final_results("hw2/results/nn1results.txt")
	# # nn1.plot_error_history()
	# print("Perceptron withen hidden nodes ", hidden_nodes1, " had a final training accuracy of ", p1_training_accuracy,
	# 	  " and a test",
	# 	  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")
	# nn1.display_prediction_history()

	"""nn experiment 1  - hidden node count = 50, momentum = 0.9"""

	# nn2 = NeuralNet(hidden_nodes2, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# nn2_epochs_ran, nn2_training_accuracy, nn2_test_accuracy, nn2_confusion_mat = nn2.run()
	# nn2.plot_accuracy_history("hw2/results/nn2.png")
	# nn2.save_final_results("hw2/results/nn2results.txt")
	# # nn2.plot_error_history()
	# print("Perceptron withen hidden nodes ", hidden_nodes2, " had a final training accuracy of ", nn2_training_accuracy, " and a test",
	# 	  " accuracy of ", nn2_test_accuracy, "after ", nn2_epochs_ran, " epochs")
	# nn2.display_prediction_history()

	"""nn experiment 1  - hidden node count = 10, momentum = 0.9"""
	# nn3 = NeuralNet(hidden_nodes3, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# nn3_epochs_ran, nn3_training_accuracy, nn3_test_accuracy, nn3_confusion_mat = nn3.run()
	# nn3.plot_accuracy_history("hw2/results/nn3.png")
	# nn3.save_final_results("hw2/results/nn3results.txt")
	# # nn3.plot_error_history()
	# print("Perceptron withen hidden nodes ", hidden_nodes3, " had a final training accuracy of ", nn3_training_accuracy,
	# 	  " and a test",
	# 	  " accuracy of ", nn3_test_accuracy, "after ", nn3_epochs_ran, " epochs")
	# nn3.display_prediction_history()
	"""Experiment 2: Vary momentum"""

	"""nn experiment 2: momentum = 0, hidden node count = 100 """
	# nn4 = NeuralNet(hidden_nodes_hundred, learning_rate, momentum_zero, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# nn4_epochs_ran, nn4_training_accuracy, nn4_test_accuracy, nn4_confusion_mat = nn4.run()
	# nn4.plot_accuracy_history("hw2/results/nn4.png")
	# nn4.save_final_results("hw2/results/nn4results.txt")
	# # nn4.plot_error_history()
	# print("Perceptron with momentum " , momentum_zero, "and 100 hidden nodes had a final training accuracy of ",
	# nn4_training_accuracy,  " and a test accuracy of ", nn4_test_accuracy, "after ", nn4_epochs_ran, " epochs")
	# nn4.display_prediction_history()

	# """nn experiment 2: momentum = 0.25, hidden node count = 100 """
	# nn5 = NeuralNet(hidden_nodes_hundred, learning_rate, momentum_quartile, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# nn5_epochs_ran, nn5_training_accuracy, nn5_test_accuracy, nn5_confusion_mat = nn5.run()
	# nn5.plot_accuracy_history("hw2/results/nn5.png")
	# nn5.save_final_results("hw2/results/nn5results.txt")
	# # nn5.plot_error_history()
	# print("Perceptron with momentum " , momentum_quartile, "and 100 hidden nodes had a final training accuracy of ",
	# nn5_training_accuracy,  " and a test accuracy of ", nn5_test_accuracy, "after ", nn5_epochs_ran, " epochs")
	# nn5.display_prediction_history()

	# # """nn experiment 2: momentum = 0.5, hidden node count = 100 """
	# nn6 = NeuralNet(hidden_nodes_hundred, learning_rate, momentum_half, data_class_count, training_data, training_data_size,
	# 				training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)
	#
	# nn6_epochs_ran, nn6_training_accuracy, nn6_test_accuracy, nn6_confusion_mat = nn6.run()
	# nn6.plot_accuracy_history("hw2/results/nn6.png")
	# nn6.save_final_results("hw2/results/nn6results.txt")
	# # nn5.plot_error_history()
	# print("Perceptron with momentum " , momentum_half, "and 100 hidden nodes had a final training accuracy of ",
	# nn6_training_accuracy,  " and a test accuracy of ", nn6_test_accuracy, "after ", nn6_epochs_ran, " epochs")
	# nn6.display_prediction_history()

	# run_experiment(hidden_nodes_hundred, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
	# 			   training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs, "nn6")
	"""Experiment 3: vary training examples"""

	"""nn7 - experiment 3: momentum = 0.9, hidden node count = 100 """

	half_data = balanced_subset_data(training_data, training_labels, data_class_count, 0.5)
	print("half data shape: ", half_data.shape)
	print("half data:\n", half_data)
	print("Test of training data size: ", training_data.shape)


# run_experiment(hidden_nodes_hundred, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
# 			   training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs, "nn7")

def balanced_subset_data(training_data, training_labels, data_class_count, desired_data_percentage):
	"""
		creates a subset of data examples where data is approximately balanced
	"""
	desired_entries_count = math.floor(training_data.shape[0] * desired_data_percentage)
	items_needed_per_class = math.floor(desired_entries_count / data_class_count)
	added_examples_count = 0
	training_data = np.array(training_data)
	# np.random.shuffle(training_data) #TODO: determine if we care to have data shuffled each time
	subset = np.zeros((desired_entries_count, training_data.shape[1]))
	not_added_indices = []

	# used to keep track of how many data examples there are for each data class- ensures we get a balanced subset
	class_tracker_map = np.zeros(data_class_count)
	for i in range(training_data.shape[0]):
		if added_examples_count == desired_entries_count:
			break
		if not class_count_exceeded(class_tracker_map, items_needed_per_class, training_labels[i]):
			subset[added_examples_count] = training_data[i]
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
			added_examples_count += 1
			class_tracker_map[training_labels[j]] += 1
			j += 1

	if added_examples_count != desired_entries_count:
		raise Exception("Something went wrong! ran out of examples while creating subset. bug here!")

	if np.unique(subset, axis=0).shape[0] != desired_entries_count:
		raise Exception("Duplicate training example found!")

	print("final class representation of training data subset: ",  class_tracker_map)
	return subset

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
				   training_data_size,
				   training_labels_matrix,
				   test_data,
				   test_data_size,
				   test_labels_matrix,
				   epochs,
				   experiment_name):
	nn = NeuralNet(hidden_nodes, learning_rate, momentum, data_class_count, training_data, training_data_size,
				   training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)

	nn_epochs_ran, nn_training_accuracy, nn_test_accuracy, nn_confusion_mat = nn.run()  # TODO: remove nn_confu matrix
	nn.plot_accuracy_history("hw2/results/" + experiment_name + ".png")
	nn.save_final_results("hw2/results/" + experiment_name + ".txt")
	# nn.plot_error_history()
	print("Perceptron with momentum ", momentum, "and 100 hidden nodes had a final training accuracy of ",
		  nn_training_accuracy, " and a test accuracy of ", nn_test_accuracy, "after ", nn_epochs_ran, " epochs")
	nn.display_prediction_history()
