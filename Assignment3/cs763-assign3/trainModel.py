import torch
from torch.utils.data import DataLoader
import loadModel
from dataset import Data
import Criterion


classifier = loadModel.load()
# device = 'cpu'
device = 'cuda:0'

trainLoader = DataLoader(Data(test=False), batch_size=32)
criterion = Criterion.Criterion()

epochs = 100
learning_rate = 1e-2
momentum = 0.9

for epoch in range(epochs):
    correct = 0
    count = 0
    for i, data in enumerate(trainLoader):
        # print i
        X, y = data
        classifier.clearGradParam()
        y_pred = classifier.forward(X)
        # print y_pred
        loss = criterion.forward(y_pred, y)
        gradLoss = criterion.backward(y_pred, y)
        classifier.backward(gradLoss)

        for layer in classifier.layers:
            if layer.type == 'Linear':
                layer.W_decrement = momentum * layer.W_decrement + layer.gradW
                layer.B_decrement = momentum * layer.B_decrement + layer.gradB
                layer.W -= learning_rate * layer.W_decrement
                layer.B -= learning_rate * layer.B_decrement

        label = torch.argmax(y_pred, dim=1)
        correct += torch.sum(label == y - 1).item()
        count += len(y)

    print 'Epoch', epoch, 'complete'
    print 'Training Accuracy:', correct * 100 / count, '\b%'
