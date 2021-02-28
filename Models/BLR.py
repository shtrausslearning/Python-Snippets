import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from numpy.linalg import cholesky, det, lstsq, inv, eigvalsh, pinv
from scipy.optimize import minimize

class BLR(BaseEstimator,RegressorMixin):
	
	# Calculate Posterior Prediction
	def posterior(self,X,y,alpha,beta):
	# Computes mean and covariance matrix of the posterior distribution
		S_N_inv = alpha * np.eye(X.shape[1]) + beta * X.T.dot(X)
		S_N = pinv(S_N_inv)
		m_N = beta * S_N.dot(X.T).dot(y)
		return m_N, S_N, S_N_inv
	
	# instantiation values
	def __init__(self,alpha=1.0e-5,beta=1e-5,max_iter=300,rtol=1.0e-5,verbose=False,opt=True):    
		self.max_iter = max_iter # class contains only tunable hyperparameters (max convergence iteration)
		self.rtol = rtol       # convergence tolerance for hyperparameters
		self.alpha = alpha   # hyperparameter 
		self.beta = beta     # hyperparameter
		self.verbose = verbose # callouts y/n
		self.opt = opt        # optimisation y/n
		
	def fit(self,X,y):
		
		if(self.verbose):
			print('Input Types: {type(X) {type(y)}}')
		if(type(X) is np.ndarray):
			self.X = X;self.y = y
		else:
			self.X = X.values; self.y = y.values
			
		# Hyperparameter tuning using implicit objective function parameters:
		# Maximization of the log marginal likelihood wrt/ alpha,beta
		if(self.opt is True):
			
			alpha_0 = 1e-10
			beta_0 = 1e-10
			N, M = X.shape
			eigenvalues_0 = np.linalg.eigvalsh(X.T.dot(X))
			self.beta = beta_0
			self.alpha = alpha_0
			
			for i in range(self.max_iter):
				
				beta_prev = self.beta
				alpha_prev = self.alpha
				eigenvalues = eigenvalues_0 * self.beta
				
				self.m_N, self.S_N, self.S_N_inv = self.posterior(X,y,self.alpha, self.beta)
				
				gamma = np.sum(eigenvalues / (eigenvalues + self.alpha))
				self.alpha = gamma / np.sum(self.m_N ** 2)
				beta_inv = 1.0 / (N - gamma) * np.sum((y - X.dot(self.m_N)) ** 2)
				
				self.beta = 1.0 / beta_inv
				
				if np.isclose(alpha_prev,self.alpha,rtol=self.rtol) and np.isclose(beta_prev, self.beta, rtol=self.rtol):
					if(self.verbose):
						print(f'Tolerance Reached: {self.alpha,self.beta}')
					break
				
		else:
			self.m_N, self.S_N, self.S_N_inv = self.posterior(X,y,self.alpha, self.beta)
			
		return self
	
	def predict(self,Xm):
		
		if(type(Xm) is np.ndarray):
			self.Xm = Xm
		else:
			self.Xm = Xm.values
		
		self.mu_s = Xm.dot(self.m_N)
		self.cov_s = 1.0 / self.beta + np.sum(Xm.dot(self.S_N) * Xm, axis=1)
		return self.mu_s