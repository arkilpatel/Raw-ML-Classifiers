import numpy as np
import pandas as pd

from src.decision_tree import DecisionTree

class RandomForest:
	def __init__(self,n_trees=10, max_depth=5,split_val_metric='mean',min_info_gain=0.0,split_node_criterion='gini', max_features=5, bootstrap=True, n_cores=1):
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.split_val_metric = split_val_metric
		self.min_info_gain = min_info_gain
		self.split_node_criterion = split_node_criterion
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.n_cores = n_cores
		self.trees = None
		
	def subsample(self,X_train,y_train, ratio):
		sample_X,sample_y = list(),list()
		n_sample = round(len(X_train) * ratio)
		while len(sample_X) < n_sample:
			index = np.random.randint(len(X_train))
			sample_X.append(X_train[index])
			sample_y.append(y_train[index])
		
		return sample_X,sample_y

	def train(self,X_train,y_train):
		features = []  
		cols = X_train.columns
		while len(features) < self.max_features:
			index = np.random.randint(len(cols))
			if cols[index] not in features:
				features.append(cols[index])
				
		print(features)
		X_train = X_train[features]
		
		X_train = np.array(X_train.values)
		y_train = y_train.values
		trees = list()
		sample_size = np.random.rand()
		for i in range(self.n_trees):
			if self.bootstrap:
				sample_X,sample_y = self.subsample(X_train,y_train, sample_size)
			else:
				sample_X,sample_y = X_train,y_train
				
			sample_X,df = pd.DataFrame(sample_X),pd.DataFrame(sample_y)
			sample_y = df[0]
			#print(sample_y.values)

			dt = DecisionTree(self.max_depth,self.split_val_metric,self.min_info_gain,self.split_node_criterion)
			tree = dt.build_tree(sample_X, sample_y)
			trees.append(tree)

		self.trees = trees
		return self.trees
	
	def fit_predict(self,X_train,y_train,X_test):
		#features = []  
		#cols = X_train.columns
		#while len(features)!=len(cols) and len(features) < self.max_features:
			#print(len(features))
			#index = np.random.randint(len(cols))
			#if cols[index] not in features:
				#features.append(cols[index])
				
		#print(features)
		#X_train = X_train[features]
		#X_train = np.array(X_train.values)
		#y_train = y_train.values
		y_train_new = y_train.values
		trees = list()
		cols = X_train.columns
		sample_size = np.random.rand()
		predictions = []
		for i in range(self.n_trees):
			
			if self.bootstrap:
				features = []
				i=0
				while len(features)!=len(cols) and len(features) < self.max_features:
					#print(len(features))
					index = np.random.randint(len(cols))
					if cols[index] not in features:
						features.append(cols[index])
					if len(features)>=1 and i == self.max_features+2:
						break
					i+=1

				#print(features)
				X_train_new = X_train[features]
				X_train_new = np.array(X_train_new.values)
			
				sample_X,sample_y = self.subsample(X_train_new,y_train_new, sample_size)
				sample_X,df = pd.DataFrame(sample_X),pd.DataFrame(sample_y)
				sample_y = df[0]
			else:
				sample_X,sample_y = X_train,y_train
				
			
			#print(sample_y.values)

			dt = DecisionTree(self.max_depth,self.split_val_metric,self.min_info_gain,self.split_node_criterion)
			dt.build_tree(sample_X, sample_y)
			prediction = dt.predict_new(X_test)
			predictions.append(prediction)
			
		#print(np.array(predictions).shape)
		
		df = np.array(predictions)
		print(df.shape)
		df = pd.DataFrame(df)
		final_predictions = df.mode(axis = 0).values[0]
		
		return final_predictions

	def predict_RF(self,X_test):
		dt = DecisionTree(self.max_depth,self.split_val_metric,self.min_info_gain,self.split_node_criterion)
		predictions = list()
		for tree in trees:
			prediction_t = dt.predict_util(tree,X_test)
			print(np.array(prediction_t).shape)
			predictions.append(prediction_t)
		
		print(predictions)
		df = np.array(predictions)
		print(df.shape)
		df = pd.DataFrame(df)
		df.mode(axis = 0)
		final_predictions = np.array(df.mode(axis = 0).values)[0]

		return final_predictions