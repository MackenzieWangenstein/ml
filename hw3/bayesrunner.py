import pandas as pd
import numpy as np
import math
import pprint

from hw3.bayesclassifier import BayesClassifier


def run():
	data = np.array(pd.read_csv("hw3/spambase.csv"))  # dataset.values returns an np array
	print("data.shape ", data.shape)
	_data_spam = data[data[:, 57] > 0, :]
	print(_data_spam[_data_spam[:, 56] == 278, :])
	# _data_spam = _data_spam[:, 0:_data_spam.shape[1]-1]


	_data_not_spam = data[data[:, 57] <= 0, :]
	# _data_not_spam = _data_not_spam[:, 0:_data_not_spam.shape[1]-1]
	np.random.shuffle(_data_spam)
	np.random.shuffle(_data_not_spam)

	spam_split_ind = math.floor(_data_spam.shape[0] / 2)
	spam_train, spam_test = _data_spam[:spam_split_ind, ], _data_spam[spam_split_ind:]
	print("shape of spam_train: \n", spam_train[0])
	print("shape of spam_test: ", spam_test.shape)

	not_spam_ind = math.floor(_data_not_spam.shape[0] / 2)
	not_spam_train, not_spam_test = _data_not_spam[:not_spam_ind], _data_not_spam[not_spam_ind:]
	print("shape of not spam_train: ", not_spam_train.shape)
	print("shape of  not spam_test: ", not_spam_test.shape)

	bc = BayesClassifier(spam_train, spam_test, not_spam_train, not_spam_test)
	results, confusion_matrix, accuracy, invalid_predictions = bc.predict()
	print("accuracy: ", accuracy)
	print("\n", confusion_matrix)
	print("Printing invalid predictions")
	print(invalid_predictions)
	# print_results(results)


def test_means_and_dev_calc():
	"""Small test of that means and std deviations calculates that can be
	verified  by hand"""
	# Find the standard deviations
	# test of matrix ops
	# [3.0, 4.1, 7.2], [5.1, 6.3, 9.8]
	test_features_matrix = np.array([[3.0, 5.1], [4.1, 6.3], [7.2, 9.8]])
	print("test \n", test_features_matrix)
	print("test shape: ", test_features_matrix.shape)
	mean_spam = np.sum(test_features_matrix, axis=0) / test_features_matrix.shape[0]
	print("test of mean\n", mean_spam)
	mean_spam_matrix = np.repeat(mean_spam[np.newaxis, :], test_features_matrix.shape[0], 0)
	print("test of means matrix \n", mean_spam_matrix)

	# combined into two lines
	std_dev_inner = np.square(test_features_matrix - mean_spam_matrix)
	std_dev = np.sqrt(std_dev_inner.sum(axis=0) / test_features_matrix.shape[0])
	print("two steps ", std_dev)

	not_spam_test_features_matrix = np.array([[2.0, 1.1], [4.1, 2.0], [8.1, 9.4]])
	not_mean_spam = np.sum(not_spam_test_features_matrix, axis=0) / not_spam_test_features_matrix.shape[0]
	not_mean_spam_matrix = np.repeat(not_mean_spam[np.newaxis, :], not_spam_test_features_matrix.shape[0], 0)
	not_std_dev_inner = np.square(not_spam_test_features_matrix - not_mean_spam_matrix)
	not_std_dev = np.sqrt(not_std_dev_inner.sum(axis=0) / not_spam_test_features_matrix.shape[0])

	print("test of not spam mean\n", not_mean_spam)
	print("test of not spam std ", not_std_dev)
	print("test features matrix\n", test_features_matrix)

	col_labels = np.ones((test_features_matrix.shape[0], 1))
	test_features_matrix = np.append(test_features_matrix, col_labels, axis=1)
	print("test features matrix\n", test_features_matrix)
	col_labels_zero =  np.zeros((not_spam_test_features_matrix.shape[0], 1))
	not_spam_test_features_matrix = np.append(not_spam_test_features_matrix,col_labels_zero, axis=1)
	print("testnot spam features matrix\n", not_spam_test_features_matrix)


def print_results(results_dict):
	pprint.pprint(results_dict)
