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
	momentum = 0.9
	epochs = 1
	# epochs = 50
	hidden_nodes1 = 20
	hidden_nodes2 = 50
	hidden_nodes3 = 100

	training_labels_matrix = np.full((training_data_size, data_class_count), 0.1)
	for i in range(training_data_size):
		_train_target_output = training_labels[i]
		training_labels_matrix[i][_train_target_output] = 0.9

	test_labels_matrix = np.full((test_data_size, data_class_count), 0.1)
	for j in range(test_data_size):
		_test_target_output = test_labels[j]
		test_labels_matrix[j][_test_target_output] = 0.9


	nn1 = NeuralNet(hidden_nodes1, learning_rate, momentum, data_class_count, training_data, training_data_size,
					training_labels_matrix, test_data, test_data_size, test_labels_matrix, epochs)

	p1_epochs_ran, p1_training_accuracy, p1_test_accuracy, p1_confusion_mat = nn1.run()
	# nn1.plot_accuracy_history()
	# nn1.plot_error_history()

	print("Perceptron withen hidden nodes ", hidden_nodes1, " had a final training accuracy of ", p1_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")
	nn1.display_prediction_history()
	print("test confusion matrix for hidden node count: ", hidden_nodes1)
	print(p1_confusion_mat)

