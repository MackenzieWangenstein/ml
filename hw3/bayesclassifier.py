import numpy as np
import math
import perceptronutility as putil


class BayesClassifier(object):
	def __init__(self,
				 spam_train,
				 spam_test,
				 not_spam_train,
				 not_spam_test):
		self.feature_count = spam_train.shape[1] - 1
		self.spam_train = spam_train[:, 0:self.feature_count]  #[:, 0:_data_spam.shape[1]-1]
		self.spam_train_count = spam_train.shape[0]
		self.spam_test = spam_test[:, 0:self.feature_count]
		self.not_spam_train = not_spam_train[:, 0:self.feature_count]
		self.not_spam_train_count = not_spam_train.shape[0]
		self.not_spam_test = not_spam_test[:, 0:self.feature_count]
		self.prob_spam = self.spam_train_count / (self.spam_train_count + self.not_spam_train_count)
		self.prob_not_spam = self.not_spam_train_count / (self.spam_train_count + self.not_spam_train_count)
		self.spam_means = []
		self.spam_std_devs = []
		self.not_spam_means = []
		self.not_spam_std_devs = []
		self.test_set_labeled = np.concatenate((spam_train, not_spam_train), axis=0)
		self.test_set = self.test_set_labeled[:,0:self.feature_count]
		self.test_confusion_matrix = np.zeros((2,2))


	def predict(self):
		# PART 2
		self.calc_means_and_std_devs()
		print("spam mean for 57 features ", self.spam_means)
		print("spam std dev for ", self.spam_std_devs.shape, " features: ", self.spam_std_devs)
		print("not spam mean for 57 features ", self.not_spam_means)
		print("not spam std dev for 57 features: ", self.not_spam_std_devs)

		test_prediction_results, invalid_predictions = self.predict_class()
		_test_accuracy = putil.compute_accuracy(self.test_confusion_matrix)
		return test_prediction_results, self.test_confusion_matrix, _test_accuracy, invalid_predictions

	def predict_class(self):
		#create a dict to hold the probability of spam, not spam, and class predicted
		test_prediction_results = dict()
		invalid_predictions = dict()
		for data_idx in range(self.test_set_labeled.shape[0]):
			spam_feature_probs = np.zeros(self.feature_count) #holds P(x1 | spam)...P(xn | spam)
			not_spam_feature_probs = np.zeros(self.feature_count) #holds P(x1 | not_spam)...P(xn | not_spam)
			for feature_pos in range(self.feature_count):
				x_sub = self.test_set_labeled[data_idx][feature_pos]
				spam_feature_probs[feature_pos] = self.gaussian_naive_bayes_alg(x_sub,
																					self.spam_means[feature_pos],
																					self.spam_std_devs[feature_pos])
				not_spam_feature_probs[feature_pos] = self.gaussian_naive_bayes_alg(x_sub,
																					self.not_spam_means[feature_pos],
																					self.not_spam_std_devs[feature_pos])
			prob_of_spam = math.log(self.prob_spam)+np.sum(spam_feature_probs)
			prob_of_not_spam = math.log(self.prob_not_spam)+np.sum(not_spam_feature_probs)

			_predicted = 0
			if prob_of_spam > prob_of_not_spam:
				_predicted = 1



			_actual = int(self.test_set_labeled[data_idx][self.feature_count])
			self.test_confusion_matrix[_predicted][_actual] += 1
			test_prediction_results[data_idx] = {
				'prob_spam': prob_of_spam,
				'prob_not_spam': prob_of_not_spam,
				'predicted': _predicted ,
				'actual': _actual}

			if _actual != _predicted:
				invalid_predictions[data_idx]  = {
					'prob_spam': prob_of_spam,
					'prob_not_spam': prob_of_not_spam,
					'predicted': _predicted,
					'actual': _actual
				}
		return test_prediction_results, invalid_predictions

	def calc_means_and_std_devs(self):
		# get mean and std devations for spam features

		# for each featurea add up all rows and divide by spam count
		self.spam_means = np.sum(self.spam_train, axis=0) / self.spam_train_count
		_mean_spam_matrix = np.repeat(self.spam_means[np.newaxis, :], self.spam_train.shape[0],0)  # repeat mean row
		spam_std_dev_inner = np.square(	self.spam_train - _mean_spam_matrix)  # (fi - mi)^2 4each feat:shape[data,feat]
		self.spam_std_devs = np.sqrt(	spam_std_dev_inner.sum(axis=0) / self.spam_train_count)  #std dev 4each feature
		self.spam_std_devs[self.spam_std_devs == 0] = 0.0001

		self.not_spam_means = np.sum(self.not_spam_train, axis=0) / self.not_spam_train_count  #
		not_mean_spam_matrix = np.repeat(self.not_spam_means[np.newaxis, :], self.not_spam_train.shape[0],0)
		not_spam_std_dev_inner = np.square(self.not_spam_train - not_mean_spam_matrix)
		self.not_spam_std_devs = np.sqrt(not_spam_std_dev_inner.sum(axis=0) / self.not_spam_train_count)
		self.not_spam_std_devs[self.not_spam_std_devs == 0] = 0.0001

	def gaussian_naive_bayes_alg(self, x_sub, mean, std_dev):
		"""

		:param x_sub:
		:param mean:
		:param std_dev:
		:return: log(P(x_sub_i | class_sub_j)
		"""
		scalar = 1 / (math.sqrt(2 * math.pi) * std_dev)
		inner = (math.pow(x_sub - mean, 2) / (2 * pow(std_dev, 2)))
		exp_eval = math.exp(-inner)
		if exp_eval == 0:
			exp_eval = np.nextafter(1/math.pow(inner,13), 1)
		return math.log(scalar* exp_eval)