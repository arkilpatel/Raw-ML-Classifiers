import numpy as np
import pandas as pd

class Logistic_Regression:
	def __init__(self, regulariser = 'L2', lamda = 0, num_steps = 100000, learning_rate = 0.01, initial_wts = None, verbose=False):
		self.regulariser = regulariser
		self.lamda = lamda
		self.num_steps = num_steps
		self.learning_rate = learning_rate
		self.initial_wts = initial_wts
		self.verbose = verbose
	
	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
	
	def __loss(self, h, y, wt, reg):
		epsilon = 1e-5
		if reg == 'L2':
			return (np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)) + ((self.lamda/2) * np.dot(wt.T,wt)))/y.size
		else:
			return (np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)) + ((self.lamda) * np.sum(np.absolute(wt))))/y.size

	def __sgn_matrix(self, W):
		ls = []
		for i in W:
			if i>0:
				ls.append(1)
			elif i<0:
				ls.append(-1)
			else:
				ls.append(0)
		ls = np.asarray(ls)
		return ls
	
	def __generate_random(self):
		list1 = np.random.rand(1)
		for i in list1:
			num1 = i
		return num1
	
	def __weight_generator(self, X, initial_wts):
		if initial_wts == None:
			theta_temp = np.random.rand(X.shape[1])
			bias_temp = self.__generate_random()
		else:
			theta_temp = []
			for i in range(1, initial_wts.size):
				theta_temp.append(initial_wts[i])
			theta_temp = np.asarray(theta_temp)
			bias_temp = initial_wts[0]
		return theta_temp, bias_temp
	
	def __binary_logreg(self, X, y, theta_par, bias_par):
		theta = theta_par
		bias = bias_par
		for i in range(self.num_steps):
			z = np.dot(X, theta) + bias
			h = self.__sigmoid(z)
			
			if self.regulariser == 'L2':
				gradient_w = (np.dot(X.T, (h - y))+(self.lamda*theta)) / y.size
			else:
				gradient_w = (np.dot(X.T, (h - y))+(self.lamda*self.__sgn_matrix(theta))) / y.size
				
			gradient_b = np.sum(h-y)/y.size

			theta -= self.learning_rate * gradient_w
			bias -= self.learning_rate * gradient_b

			z = np.dot(X, theta) + bias
			h = self.__sigmoid(z)
			loss1 = self.__loss(h, y, theta, self.regulariser)
				
			if(self.verbose ==True and i % 1000 == 0):
				print('loss after'+ str(i) + ': ' + str(loss1) +' \t')   
			
		return theta, bias
	
	def fit(self, X, y):
		self.theta, self.bias = self.__weight_generator(X, self.initial_wts)  

		self.classes_ = np.array(sorted(list(set(y))))
		self.n_classes_ = len(self.classes_)
		
		if self.n_classes_ == 2:
			self.theta_, self.bias_ = self.__binary_logreg(X, y, self.theta_, self.bias_)
		else:
			self.para_theta_ = {}
			self.para_bias_ = {}
			for i in self.classes_:
				y_new = np.copy(y)
				for j in range(y.size):
					if y[j] == i:
						y_new[j] = 1
					else:
						y_new[j] = 0
				
			theta1, bias1 = self.__binary_logreg(X, y_new, self.theta, self.bias)
			self.para_theta_[i] = np.copy(theta1)
			self.para_bias_[i] = np.copy(bias1)  
	
	def predict_prob(self, X):
		if self.n_classes_ == 2: 
			return self.__sigmoid(np.dot(X, self.theta) + self.bias).round()
		else:
			self.hypothesis_ = {}
			mx = []
			for i in self.classes_:
				self.hypothesis_[i] = self.__sigmoid(np.dot(X, self.para_theta_[i]) + self.para_bias_[i])
				mx.append(self.hypothesis_[i])
			df = pd.DataFrame(mx)
			search_index = df.idxmax().values
			predictions = [self.classes_[search_index[i]] for i in range(len(search_index))]
			return predictions
	
	def predict(self, X):
		return self.predict_prob(X)