import numpy as np

from src.decision_tree import DecisionTree

class Stacking:
	def __init__(self, tuples = [(DecisionTree(), 2)]):
		self.tuples = tuples

	def cross_validation_split(self, X,y, folds=3):
		dataset_split = list()
		dataset_splity = list()
		dataset_copy = list(X)
		dataset_copyy = list(y)
		fold_size = int(len(X) / folds)
		for i in range(folds):
			fold = list()
			foldy = list()
			while len(fold) < fold_size:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy)
				foldy.append(dataset_copyy)
			dataset_split.append(fold)
			dataset_splity.append(foldy)
		
		fold_size = len(X) - int(float(len(X))/float(folds))
		
		X_split,y_split = X.iloc[:fold_size],y.iloc[:fold_size]
		dataset_split = X_split
		dataset_splity = y_split
		#print(len(dataset_split),len(dataset_splity))
		return dataset_split, dataset_splity

	def fit(self,X_train, X_test, y_train):  
		Predictions = []
		for i in range(len(self.tuples)): 
			Xtrain, ytrain = self.cross_validation_split(X_train, y_train, folds=self.tuples[i][1])
			#print(np.array(Xtrain).shape)
			for j in range(self.tuples[i][1]): 
				if isinstance(self.tuples[i][0], DecisionTree):
					ytrain_new = ytrain[0]
					self.tuples[i][0].build_tree(Xtrain , ytrain_new)
				elif isinstance(self.tuples[i][0], RandomForest):
					preds = self.tuples[i][0].fit_predict(Xtrain , ytrain, X_test)
				else:
					self.tuples[i][0].fit(Xtrain_list , ytrain_list)
				
				if isinstance(self.tuples[i][0], DecisionTree):
					preds=self.tuples[i][0].predict_new(X_test)
				elif isinstance(self.tuples[i][0], RandomForest):
					preds=self.tuples[i][0].fit_predict(Xtrain,ytrain,X_test)
				else:
					preds=self.tuples[i][0].predict(X_test)
				
				Predictions.append(preds)
		return Predictions

	def predict(self,Predictions):
		pred = stats.mode(Predictions)
		pred = np.array(pred.mode[0])
		return pred