import pandas as pd
import numpy as np
import math


def run():
	data = np.array(pd.read_csv("hw3/spambase.csv"))  # dataset.values returns an np array
	print("data.shape ", data.shape)
	_data_spam = data[data[:, 57] > 0, :]
	_data_not_spam = data[data[:, 57] <= 0, :]
	np.random.shuffle(_data_spam)
	np.random.shuffle(_data_not_spam)

	spam_split_ind = math.floor(_data_spam.shape[0] / 2)
	spam_train, spam_test = _data_spam[:spam_split_ind, ], _data_spam[spam_split_ind:]
	spam_train_count = spam_train.shape[0]
	print("shape of spam_train: ", spam_train.shape)
	print("shape of spam_test: ", spam_test.shape)

	not_spam_ind = math.floor(_data_not_spam.shape[0] / 2)
	not_spam_train, not_spam_test = _data_not_spam[:not_spam_ind], _data_not_spam[not_spam_ind:]
	not_spam_train_count = not_spam_train.shape[0]
	print("shape of not spam_train: ", not_spam_train.shape)
	print("shape of  not spam_test: ", not_spam_test.shape)

	# Class Probabilities
	prob_spam = spam_train_count / (spam_train_count + not_spam_train_count)
	print("Prob of spam: ", prob_spam)
	prob_not_spam = not_spam_train_count / (spam_train_count + not_spam_train_count)
	print("Prob of not spam: ", prob_not_spam)

	# get mean and std devations for spam features
	mean_spam = np.sum(spam_train, axis=0) / spam_train_count  #for each featurea add up all rows and divide by spam count
	mean_spam_matrix = np.repeat(mean_spam[np.newaxis, :],spam_train.shape[0], 0) #reapet mean row to match shape of spam
	spam_std_dev_inner = np.square(spam_train - mean_spam_matrix) #get (fi - mi)^2 for each feature: shape[data, feature]
	spam_std_dev = np.sqrt(spam_std_dev_inner.sum(axis=0)/spam_train_count) #get std dev for each feature for spam
	spam_std_dev[spam_std_dev < 1] = 0.0001
	print("spam mean for 58 features ", mean_spam)
	print("spam std dev for 58 features: ", spam_std_dev)

	#get mean and std devations for not spam features
	not_mean_spam = np.sum(not_spam_train, axis=0) / not_spam_train_count  #for each featurea add up all rows and divide by spam count
	not_mean_spam_matrix = np.repeat(not_mean_spam[np.newaxis, :],not_spam_train.shape[0], 0) #reapet mean row to match shape of spam
	not_spam_std_dev_inner = np.square(not_spam_train - not_mean_spam_matrix) #get (fi - mi)^2 for each feature: shape[data, feature]
	not_spam_std_dev = np.sqrt(not_spam_std_dev_inner.sum(axis=0)/not_spam_train_count) #get std dev for each feature for spam
	not_spam_std_dev[not_spam_std_dev < 1] = 0.0001
	print("spam mean for 58 features ", not_mean_spam)
	print("spam std dev for 58 features: ", not_spam_std_dev)


def testPart2():

	# Find the standard deviations

	# test of matrix ops

	# [3.0, 4.1, 7.2], [5.1, 6.3, 9.8]
	test_features_matrix = np.array([[3.0, 5.1], [4.1, 6.3], [7.2, 9.8]])
	print("test \n", test_features_matrix)
	print("test shape: ", test_features_matrix.shape)
	# print("test of sum ", np.sum(test_features_matrix, axis=0))
	mean_spam = np.sum(test_features_matrix, axis=0) / test_features_matrix.shape[0]
	print("test of mean\n", mean_spam)
	# mean_test_matrix = np.broadcast_to(mean_test, (2, test_features_matrix[0]))
	mean_spam_matrix = np.repeat(mean_spam[np.newaxis, :],test_features_matrix.shape[0], 0)
	# mean_test_matrix = np.array([mean_test,]*test_features_matrix[0])
	print("test of means matrix \n", mean_spam_matrix)

	#combined into two lines
	std_dev_inner = np.square(test_features_matrix - mean_spam_matrix)
	std_dev = np.sqrt(std_dev_inner.sum(axis=0)/test_features_matrix.shape[0])
	print("two steps ", std_dev)





