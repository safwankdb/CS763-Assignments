import torch
import numpy as np
from math import sqrt

device = 'cpu'

"""
Currently implemented without bias, generally ignored in RNNs.
"""

class RNN():
	def __init__(self, D, H):
		"""
		Y->number of outputs=2
		D-> word vector size, H-> hidden layer size
		self.layers stores all the hidden values through the sequence(it is an array of tensors)
		"""
		self.D=D
		self.H=H
		self.Whh = torch.randn(H, H).double().to(device) / sqrt(H)
		self.Wxh = torch.randn(D, H).double().to(device) / sqrt(D)
		self.Why = torch.randn(H, 2).double().to(device) / sqrt(H)
		# self.Bh = torch.zeros(1, H).double().to(device)
		# self.By = torch.zeros(1, 2).double().to(device)
		self.gradWhh=torch.zeros_like(self.Whh).to(device)
		self.gradWxh=torch.zeros_like(self.Wxh).to(device)
		self.gradWhy=torch.zeros_like(self.Why).to(device)
		

	def forward(self, input):
		"""
		T-> number of layers the RNN unfolds into
		input dimensions are T*batch_size*D i.e. one hot encoding
		h'= h*Whh+x*Wxh
		"""
		self.input=input
		self.batch_size = input.shape[1]
		self.T=input.shape[0]
		self.hidden=[torch.zeros(self.batch_size,self.H).double().to(device)] #list of hidden states
		for i in range(0,self.T):
			self.hidden.append(torch.tanh(torch.matmul(self.hidden[-1], self.Whh)\
				+torch.matmul(self.input[0], self.Wxh)\
				# +torch.matmul(torch.ones(self.batch_size, 1).double().to(device), self.Bh)
				).to(device))
		self.out=torch.matmul(self.hidden[-1], self.Why)
		return self.out

	def reset(self):
		"""
		Need to call this before every backpropagation
		"""
		self.gradWhh=torch.zeros_like(self.Whh)
		self.gradWxh=torch.zeros_like(self.Wxh)
		self.gradWhy=torch.zeros_like(self.Why)   
		# self.gradBh=torch.zeros_like(self.Why).to(device)   


	def backward(self, gradOut):
		"""
		gradOut has dimensions batch_size*2
		curGrad has dimensions batch_size*hidden
		"""
		curGrad=torch.matmul(gradOut, self.Why.t())
		self.gradWhy=torch.matmul(self.hidden[-1].t(), gradOut)
		for i in range(self.T-1, 0, -1):
			curGrad=self.ones(self.batch_size, self.H)-torch.dot(curGrad, curGrad)
			self.gradWhh += torch.matmul(self.hidden[i-1].t(), curGrad)
			self.gradWxh += torch.matmul(self.input[i].t(), curGrad)
			# self.gradBh = torch.sum(gradOutput, dim=0).view(-1, 1)

