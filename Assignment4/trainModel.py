import torch
import loadModel
from dataset import Data
import Criterion
from RNN import RNN
from Model import Model

""" Currently implemented without momentum """
device = 'cuda:0'

trainData = Data(test=False)
criterion = Criterion.Criterion()
layer=RNN(154,128,2)
model=Model(layer)

batch_size = 128
epochs = 100
alpha = 0.1 #generally use high Learning rate in RNN since vanishing gradients

for epoch in range(epochs):
    correct = 0
    count = 0
    for i in range(0, trainData.m, batch_size):
        # print i
        X, y = trainData.sample(batch_size, i)
        classifier.reset()
        y_pred = classifier.forward(X)
        # print y_pred
        loss = criterion.forward(y_pred, y)
        gradLoss = criterion.backward(y_pred, y)
        classifier.backward(gradLoss)

        layer.Whh -= alpha * layer.gradWhh
        layer.Wxh -= alpha * layer.gradWxh
        layer.Why -= alpha * layer.gradWhy

        label = torch.argmax(y_pred, dim=1)
        correct += torch.sum(label == y - 1).item()
        count += len(y)

    print 'Epoch', epoch, 'complete'
    print 'Training Accuracy:', correct * 100 / count, '\b%'
    if (1 + epoch) % 10 == 0:
        torch.save(classifier, modelName + '/model')
