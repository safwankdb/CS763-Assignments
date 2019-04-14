import torch
import numpy as np
from math import sqrt

device = 'cpu'

class RNN():
	def __init__(self, D, H):
		"""
		D-> word vector size, H-> hidden layer size
		self.layers stores all the hidden values through the sequence(it is an array of tensors)
		"""
		self.D=D
		self.H=H
		self.Whh = torch.randn(H, H).double().to(device) / sqrt(H)
		self.Wxh = torch.randn(D, H).double().to(device) / sqrt(D)
		self.Why = torch.randn(H, 2).double().to(device) / sqrt(H)
		self.Bh = torch.zeros(1, H).double().to(device)
		self.By = torch.zeros(1, 2).double().to(device)
		self.gradWhh=torch.zeros_like(self.Whh).to(device)
		self.gradWxh=torch.zeros_like(self.Wxh).to(device)
		self.gradWhy=torch.zeros_like(self.Why).to(device)
		self.gradBh=torch.zeros_like(self.Bh).to(device)
		self.gradBy=torch.zeros_like(self.By).to(device)


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
		# print("INPUT", self.input)
		# print("Whh",self.Whh)
		for i in range(0,self.T):
			# print("HIDDEN", self.hidden[-1])
			mask=input[i,:,0]
			mask=mask.view(-1,1)
			mask=mask.repeat(1,self.H).double()
			var=torch.tanh(torch.matmul(self.hidden[-1], self.Whh)\
				+torch.matmul(self.input[i], self.Wxh)\
				+torch.matmul(torch.ones(self.batch_size, 1).double().to(device), self.Bh)\
				).to(device)
			self.hidden.append(mask*self.hidden[-1]+(1-mask)*var)
		self.out=torch.matmul(self.hidden[-1], self.Why)+torch.matmul(torch.ones(self.batch_size,1).double().to(device), self.By)
		return self.out

	def reset(self):
		"""
		Need to call this before every backpropagation
		"""
		self.gradWhh=torch.zeros_like(self.Whh)
		self.gradWxh=torch.zeros_like(self.Wxh)
		self.gradWhy=torch.zeros_like(self.Why)   
		self.gradBh=torch.zeros_like(self.Bh)
		self.gradBy=torch.zeros_like(self.By)


	def backward(self, gradOut):
		"""
		gradOut has dimensions batch_size*2
		curGrad has dimensions batch_size*hidden
		"""
		topGrad=torch.matmul(gradOut, self.Why.t()).double()
		self.gradBy+=torch.sum(gradOut, 0)
		self.gradWhy=torch.matmul(self.hidden[-1].t(), gradOut)
		curGrad=topGrad
		prevmask=torch.ones(self.batch_size, self.H).double()
		for i in range(len(self.hidden)-2, -1, -1):
			mask=self.input[i,:,0]
			mask=mask.view(-1,1)
			mask=mask.repeat(1,self.H).double()
			curGrad=(1-self.hidden[i+1]*self.hidden[i+1])*(curGrad*(1-prevmask)+topGrad*(1-mask)*prevmask)
			self.gradBh+=torch.sum(curGrad,0)
			self.gradWhh += torch.matmul(self.hidden[i].t(), curGrad)
			self.gradWxh += torch.matmul(self.input[i].t(), curGrad)
			curGrad=torch.matmul(curGrad, self.Whh.t())
			prevmask=mask
			# self.gradBh = torch.sum(gradOutput, dim=0).view(-1, 1)
