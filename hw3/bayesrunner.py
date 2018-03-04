import pandas as pd
import numpy as np
import math


def run():
	data = np.array(pd.read_csv("hw3/spambase.csv"))  # dataset.values returns an np array
	# _training_data_spam = data.iloc[:, 1:57].values  #57 or 56?

	print("data.shape ", data.shape)
	_data_spam = data[data[:, 57] > 0, :]
	_data_not_spam = data[data[:, 57] <= 0, :]

	np.random.shuffle(_data_spam)
	np.random.shuffle(_data_not_spam)



	# print(_data_spam[:,57])
	# print(_data_not_spam[:,57])
	print("test of spam set\n", _data_spam.shape)
	print("test of not spam set\n", _data_not_spam.shape)
	spam_split_ind = math.floor(_data_spam.shape[0] / 2)
	print("test of spam split ind ", spam_split_ind)
	spam_train, spam_test = _data_spam[:spam_split_ind, ], _data_spam[spam_split_ind:]

	print("shape of spam_train: ", spam_train.shape)
	print("shape of spam_test: ", spam_test.shape)

	not_spam_ind = math.floor(_data_not_spam.shape[0]/2)
	not_spam_train, not_spam_test = _data_not_spam[:not_spam_ind], _data_not_spam[not_spam_ind:]

	print("shape of not spam_train: ", not_spam_train.shape)
	print("shape of  not spam_test: ", not_spam_test.shape)