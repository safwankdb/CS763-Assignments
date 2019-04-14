""" Run like 'python3 testModel.py >test_label.txt' """
import torch
from dataset import Data
import Criterion
from RNN import RNN
from Model import Model

device = 'cpu'

testData = Data(test=True, actualTest=True, data="data/test_data.txt", D=154)
classifier = torch.load('model')

batch_size = 8

correct = 0
count = 0

print("id,label")
cnt=0

for i in range(0, testData.m, batch_size):
	if(i+batch_size>=testData.m):
		batch_size=testData.m-i
	X = testData.sample(batch_size, i)
	X = X.permute(1, 0, 2)
	y_pred = classifier.forward(X)
	label = torch.argmax(y_pred, dim=1)
	for i in range(batch_size):
		print(cnt,label[i].item(),sep=',')
		cnt+=1
	if(batch_size==testData.m-i):
		break