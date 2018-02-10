from hw2.neuralnet import NeuralNet

import numpy as np


def run_neural_network(training_data,
					   training_data_size,
					   training_labels,
					   test_data,
					   test_data_size,
					   test_labels,
					   data_class_count):
	print("data class count", data_class_count)
	learning_rate = 0.1
	momentum_default = 0.9
	momentum_zero = 0
	momentum_quartile = 0.25
	momentum_half = 0.50
	# epochs = 4
	epochs = 50
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

	nn1 = NeuralNet(hidden_nodes1, learning_rate, momentum_default, data_class_count, training_data, training_data_size,
					training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)

	p1_epochs_ran, p1_training_accuracy, p1_test_accuracy, p1_confusion_mat = nn1.run()
	nn1.plot_accuracy_history("hw2/results/nn1.png")
	nn1.save_final_results("hw2/results/nn1results.txt")
	# nn1.plot_error_history()
	print("Perceptron withen hidden nodes ", hidden_nodes1, " had a final training accuracy of ", p1_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")
	nn1.display_prediction_history()

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

	"""Experiment 3: vary training examples"""

	"""nn experiment 3: momentum = 0.9, hidden node count = 100 """



