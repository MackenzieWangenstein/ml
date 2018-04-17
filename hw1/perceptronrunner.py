from hw1.perceptron import Perceptron
import generalutility as util
import numpy as np
import time

"""
	Permission was granted by the class instructor to share graphing utility methods across projects 
	in order to grant students the flexibility needed to spend more time investigating experimental results. 
        To create more time to perform research analysis, a save_data method implemented in generalutil.java
	is sourced from Andrew keene. Directory, file, and epoch_results definitions are also sourced in order
	to use the save_data method.
	https://github.com/andy-keene/ML/blob/master/perceptron/main.py
"""


def run_perceptron(training_data,
				   training_data_size,
				   training_labels,
				   test_data,
				   test_data_size,
				   test_labels):
	# dir of form ./save/Sat-Jan-27-13:52:05-2018
	directory = './save/{}'.format(time.ctime().replace(' ', '-'))
	file = directory + '/{}'

	# create weights matrix for training set 10 rows with 748 randomly generated weights from -0.5to 0.5
	# Wij represents weight connecting input node i to output onde j
	rand_weights = np.random.uniform(low=-0.5, high=0.5,
									  size=(training_data.shape[1], 10))  # include weight for bias


	data_class_count = rand_weights.shape[1];
	epochs = 70
	n1 = 0.001
	n2 = 0.01
	n3 = 0.1

	training_labels_matrix = np.zeros((training_data_size, 10))
	for i in range(training_data_size):
		_train_target_output = training_labels[i]
		training_labels_matrix[i][_train_target_output] = 1

	test_labels_matrix = np.zeros((test_data_size, data_class_count))
	for j in range(test_data_size):
		_test_target_output = test_labels[j]
		test_labels_matrix[j][_test_target_output] = 1

	p1 = Perceptron(n1, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)
	p2 = Perceptron(n2, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)
	p3 = Perceptron(n3, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)

	p1_epochs_ran, p1_training_accuracy, p1_test_accuracy, p1_epoch_results, p1_confusion_mat = p1.run()
	print("Perceptron of learning rate ", n1, " had a final training accuracy of ", p1_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")
	util.save_data(directory, file.format(n1), p1_epoch_results)
	print("test confusion matrix for learning rate ", n1)
	print(p1_confusion_mat)

	p2_epochs_ran, p2_training_accuracy, p2_test_accuracy, p2_epoch_results, p2_confusion_mat  = p2.run()
	print("Perceptron of learning rate ", n2, " had a final training accuracy of ", p2_training_accuracy, " and a test",
		  " accuracy of ", p2_test_accuracy, " after ", p2_epochs_ran, " epochs")
	util.save_data(directory, file.format(n2), p2_epoch_results)
	print("test confusion matrix for learning rate ", n2)
	print(p2_confusion_mat)

	p3_epochs_ran, p3_training_accuracy, p3_test_accuracy, p3_epoch_results, p3_confusion_mat  = p3.run()
	print("Perceptron of learning rate ", n3, " had a final training accuracy of ", p3_training_accuracy, " and a test",
		  " accuracy of ", p3_test_accuracy, " after ", p3_epochs_ran, " epochs")
	util.save_data(directory, file.format(n3), p3_epoch_results)
	print("test confusion matrix for learning rate ", n3)
	print(p3_confusion_mat)
