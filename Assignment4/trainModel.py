import torch
from dataset import Data
import Criterion
from RNN import RNN
from Model import Model

""" Currently implemented without momentum """
device = 'cpu'

trainData = Data(test=False, m_train=1184, D=154)
criterion = Criterion.Criterion()
layer = RNN(154, 128)
classifier = Model(layer)

batch_size = 1
epochs = 40
alpha = 0.005 # generally use high Learning rate in RNN since vanishing gradients
# 0.01 was showing good results in overfit data
for epoch in range(epochs):
    correct = 0
    count = 0
    totloss = 0
    tot2loss = 0
    if(epoch >= 7):
        alpha = 0.002
    if(epoch >= 25):
        alpha = 0.001
    for i in range(0, trainData.m, batch_size):  # CHANGED
        # print(i)
        # print("Whh", layer.Whh)
        # print("Wxh", layer.Wxh)
        # print("Why", layer.Why)

        # if(i+1)%16==0:
        #     print ('Average Loss', batch_size*totloss/16)
        #     totloss=0

        X, y = trainData.sample(batch_size, i)
        X = X.permute(1, 0, 2)

        # torch.set_printoptions(profile="full")
        classifier.reset()
        y_pred = classifier.forward(X)
        # print("Y_PRED", y_pred)
        loss = criterion.forward(y_pred, y)
        # print("CUR", loss.item())
        totloss += loss.item()
        tot2loss += loss.item()
        gradLoss = criterion.backward(y_pred, y)
        classifier.backward(gradLoss)

        layer.gradWhh = torch.clamp(layer.gradWhh, -5, +5)
        layer.gradWxh = torch.clamp(layer.gradWxh, -5, +5)
        layer.gradWhy = torch.clamp(layer.gradWhy, -5, +5)
        layer.gradBy = torch.clamp(layer.gradBy, -5, +5)
        layer.gradBh = torch.clamp(layer.gradBh, -5, +5)

        layer.Whh -= alpha * layer.gradWhh
        layer.Wxh -= alpha * layer.gradWxh
        layer.Why -= alpha * layer.gradWhy
        layer.Bh -= alpha * layer.gradBh
        layer.By -= alpha * layer.gradBy

        label = torch.argmax(y_pred, dim=1)
        correct += torch.sum(label == y.long()).item()
        count += len(y)

    print('Epoch', epoch, 'complete')
    print('Average Loss', batch_size*tot2loss/count)
    print('Training Accuracy:', correct * 100 / count, '\b%')
    if (1 + epoch) % 10 == 0:
        torch.save(classifier, 'model')
