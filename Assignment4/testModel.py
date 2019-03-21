import torch
from dataset import Data
import Criterion
from RNN import RNN
from Model import Model

""" Currently implemented without momentum """
device = 'cpu'

testData = Data(test=True)
classifier=torch.load('model')

batch_size = 4

correct = 0
count = 0
for i in range(0, testData.m, batch_size):
    X, y = testData.sample(batch_size, i)
    y_pred = classifier.forward(X)
    label = torch.argmax(y_pred, dim=1)
    correct += torch.sum(label == y - 1).item()
    count += len(y)

print ('Validation Accuracy:', correct * 100 / count, '\b%')
