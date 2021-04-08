import numpy as np
from copy import deepcopy

from src.decision_tree import DecisionTreeSW

class AdaboostClassifier:
	def __init__(self, max_depth = 1, n_trees = 10, learning_rate = 0.1):
		self.max_depth = max_depth
		self.classifier = DecisionTreeSW(self.max_depth, split_val_metric = 'mean')
		self.n_trees = n_trees
		self.learning_rate = learning_rate
		self.trees_ = list()
		self.tree_weights_ = np.zeros(self.n_trees)
		self.tree_errors_ = np.ones(self.n_trees)
	
	def compute(self, X, y, sample_weight):
		local_tree = deepcopy(self.classifier)
		
		#y = y[0]
		
		local_tree.build_tree(X, y, sample_weights=sample_weight)
		 
		y_pred = local_tree.predict_new(X)    
		
		misclassified = y != y_pred
		misclassified = np.asarray(misclassified)
		local_tree_error = np.dot(misclassified, sample_weight)/np.sum(sample_weight, axis=0)
		
		#print(local_tree_error)
		
		#if local_tree_error >= 1 - (1/self.n_classes_):
			#print("Hello")
			#return None, None, None
			
		local_tree_weight = self.learning_rate * np.log(float(1 - local_tree_error)/float(local_tree_error)) + np.log(self.n_classes_ - 1)
		
		#print(local_tree_weight)
		
		if local_tree_weight <= 0:
			return None, None, None

		sample_weight = sample_weight * np.exp(local_tree_weight * misclassified)

		sample_weight_sum = np.sum(sample_weight, axis=0)
		
		if sample_weight_sum <= 0:
			return None, None, None

		sample_weight /= sample_weight_sum

		self.trees_.append(local_tree)
		#print(len(self.trees_))

		return sample_weight, local_tree_weight, local_tree_error
	
	def fit(self, X, y):
		#y = y[0]
		list_classes = sorted(set(y.values))
		self.n_classes_ = len(list_classes)
		#list_classes = sorted(list(set(y)))
		self.classes_ = np.array(list_classes)
		#self.n_classes_ = len(self.classes_)
		self.n_samples = X.shape[0]
		
		for tree in range(self.n_trees):
			if tree == 0:
				sample_weight = np.ones(self.n_samples) / self.n_samples
			sample_weight, tree_weight, tree_error = self.compute(X, y, sample_weight)
			if tree_error == None:
				break

			self.tree_errors_[tree] = tree_error
			self.tree_weights_[tree] = tree_weight

			if tree_error <= 0:
				break

		return self
	
	def predict(self, X):
		n_classes = self.n_classes_
		self.classes_ = np.array(self.classes_)
		#print(self.classes_)
		classes = self.classes_[:, np.newaxis]
		#print(classes)
		pred = None
		
		pred = sum((tree.predict_new(X) == classes).T * w for tree, w in zip(self.trees_, self.tree_weights_))
		print(pred)

		pred = pred/self.tree_weights_.sum()
		if n_classes == 2:
			pred[:, 0] = pred[:, 0]*(-1)
			pred = pred.sum(axis=1)
			return self.classes_.take(pred > 0, axis=0)
		#print(pred)
		predictions = self.classes_.take(np.argmax(pred, axis=1), axis=0)
		#print(predictions)
		return predictions