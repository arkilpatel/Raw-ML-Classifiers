import numpy as np

class Naive_Bayes(object):
	def __init__(self, type = "Gaussian", prior = []):
		self.type = type
		self.prior = prior
	
	def fit(self, X, y):
		if((self.type).lower() == "multinomial"):
			count_sample = X.shape[0]
			separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
			if len(self.prior)==0:
				self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
			else:
				self.class_log_prior_ = self.prior
			count = np.array([np.array(i).sum(axis=0) for i in separated]) + 1.0
			self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
			return self
		if((self.type).lower() == "gaussian"):
			separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
			self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
					for i in separated])
		return self

	def _prob(self, x, mean, std):
		if((self.type).lower() == "gaussian"):
			exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
			return np.log(exponent / (np.sqrt(2 * np.pi) * std))

	def predict_log_proba(self, X):
		if((self.type).lower() == "multinomial"):
			return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
					for x in X] 
		if((self.type).lower() == "gaussian"):
			return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
					for summaries in self.model] for x in X]

	def predict(self, X):
		return np.argmax(self.predict_log_proba(X), axis=1)

	def score(self, X, y):
		return sum(self.predict(X) == y) / len(y)