import numpy as np
import math


class BayesClassifier(object):
	def __init__(self,
				 spam_train,
				 spam_test,
				 not_spam_train,
				 not_spam_test):
		self.spam_train = spam_train[:]
		self.spam_train_count = spam_train.shape[0]
		self.spam_test = spam_test
		self.not_spam_train = not_spam_train
		self.not_spam_train_count = not_spam_train.shape[0]
		self.not_spam_test = not_spam_test
		self.prob_spam = self.spam_train_count / (self.spam_train_count + self.not_spam_train_count)
		self.prob_not_spam = self.not_spam_train_count / (self.spam_train_count + self.not_spam_train_count)
		self.spam_means = []
		self.spam_std_devs = []
		self.not_spam_means = []
		self.not_spam_std_devs = []
		self.test_set = np.concatenate((self.spam_train, self.not_spam_train), axis=0)
		print("Test_set shape ", self.test_set.shape)
		self.exceptions_caught = 0
		self.none_exceptions = 0
		self.feature_count = spam_train.shape[1]
		print(self.feature_count)

	def predict(self):
		# PART 2
		self.calc_means_and_std_devs()
		print("test of std devs and means")
		print("spam mean for 57 features ", self.spam_means)
		print("spam std dev for ", self.spam_std_devs.shape, " features: ", self.spam_std_devs)
		print("spam mean for 57 features ", self.not_spam_means)
		print("spam std dev for 57 features: ", self.not_spam_std_devs)  # TODO: keep or remove

		# Part 3) Use Gaussian Naive Bayes algorith to classify test set
		self.predict_class()

	def calc_means_and_std_devs(self):
		# get mean and std devations for spam features
		self.spam_means = np.sum(self.spam_train,
								 axis=0) / self.spam_train_count  # for each featurea add up all rows and divide by spam count
		_mean_spam_matrix = np.repeat(self.spam_means[np.newaxis, :], self.spam_train.shape[0],
									  0)  # reapet mean row to match shape of spam
		spam_std_dev_inner = np.square(
			self.spam_train - _mean_spam_matrix)  # (fi - mi)^2 4each feat:shape[data, feature]
		self.spam_std_devs = np.sqrt(
			spam_std_dev_inner.sum(axis=0) / self.spam_train_count)  # get std dev for each feature for spam
		self.spam_std_devs[self.spam_std_devs == 0] = 0.0001

		# get mean and std devations for not spam features
		self.not_spam_means = np.sum(self.not_spam_train,
									 axis=0) / self.not_spam_train_count  # for each featurea add up all rows and divide by spam count
		not_mean_spam_matrix = np.repeat(self.not_spam_means[np.newaxis, :], self.not_spam_train.shape[0],
										 0)  # reapet mean row to match shape of spam
		not_spam_std_dev_inner = np.square(self.not_spam_train - not_mean_spam_matrix)
		self.not_spam_std_devs = np.sqrt(
			not_spam_std_dev_inner.sum(axis=0) / self.not_spam_train_count)  # get std dev for each feature for spam
		self.not_spam_std_devs[self.not_spam_std_devs == 0] = 0.0001

	def predict_class(self):
		#create a dict to hold the probability of spam, not spam, and class predicted
		# convert each
		print("test of shape for test set: ", self.test_set.shape)
		for data_idx in range(self.test_set.shape[0]):
			spam_feature_probs = np.zeros(self.feature_count)
			not_spam_feature_probs = np.zeros(self.feature_count)
			for feature_pos in range(self.feature_count):
				x_sub = self.test_set[data_idx][feature_pos]
				spam_feature_probs[feature_pos] = self.gaussian_naive_bayes_alg(x_sub,
																					self.spam_means[feature_pos],
																					self.spam_std_devs[feature_pos])
				not_spam_feature_probs[feature_pos] = self.gaussian_naive_bayes_alg(x_sub,
																					self.not_spam_means[feature_pos],
																					self.not_spam_std_devs[feature_pos])
				# print("norm features distrubted post", spam_feature_probs)
				# print("norm features distrubted neg", not_spam_feature_probs)
				# print("numbers of exceptions caught: ", self.exceptions_caught)
				# print("non exceptions ", self.none_exceptions)
				prob_of_spam = math.log(self.prob_spam)*np.sum(spam_feature_probs)
				# print("prob of spam: ", prob_of_spam)
				prob_of_not_spam = math.log(self.prob_not_spam)*np.sum(not_spam_feature_probs)
				# print("prob of not spam: ", prob_of_not_spam)


	def gaussian_naive_bayes_alg(self, x_sub, mean, std_dev):
		"""

		:param x_sub:
		:param mean:
		:param std_dev:
		:return: log(P(x_sub_i | class_sub_j)
		"""
		scalar = 1 / (math.sqrt(2 * math.pi) * std_dev)
		# print("scalar: ", scalar)
		inner = (math.pow(x_sub - mean, 2) / (2 * pow(std_dev, 2)))
		# print("inner: ", inner)
		exp_eval = math.exp(-inner)
		if exp_eval == 0:
			exp_eval = np.nextafter(1 / inner, 1)
		# print("exp val: ", exp_eval )
		N =  math.log(scalar *exp_eval)
		return N
