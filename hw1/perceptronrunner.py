from hw1.perceptron import Perceptron

def run_perceptron(training_data,
				   training_data_size,
				   training_labels_matrix,
				   test_data,
				   test_data_size,
				   test_labels_matrix,
				   rand_weights):
	data_class_count = 10;
	epochs = 2  # TODO: replace with 70
	n1 = 0.001
	n2 = 0.01
	n3 = 0.1

	p1 = Perceptron(n1, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)
	p2 = Perceptron(n2, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)
	p3 = Perceptron(n3, data_class_count, training_data, training_data_size, training_labels_matrix, test_data,
					test_data_size, test_labels_matrix, rand_weights, epochs)

	p1_epochs_ran, p1_training_accuracy, p1_test_accuracy = p1.run()
	print("Perceptron of learning rate ", n1, " had a final training accuracy of ", p1_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, "after ", p1_epochs_ran, " epochs")

	p2_epochs_ran, p2_training_accuracy, p2_test_accuracy = p2.run()
	print("Perceptron of learning rate ", n2, " had a final training accuracy of ", p2_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, " after ", p2_epochs_ran, " epochs")

	p3_epochs_ran, p3_training_accuracy, p3_test_accuracy = p3.run()
	print("Perceptron of learning rate ", n3, " had a final training accuracy of ", p3_training_accuracy, " and a test",
		  " accuracy of ", p1_test_accuracy, " after ", p3_epochs_ran, " epochs")
