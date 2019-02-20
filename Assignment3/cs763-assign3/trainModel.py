import torch
import loadModel
from dataset import Data
import Criterion


classifier = loadModel.load()
# device = 'cpu'
device = 'cuda:0'

trainData = Data(test=False)
criterion = Criterion.Criterion()

batch_size = 32
epochs = 200
alpha = 1e-2
momentum = 0.9

for epoch in range(epochs):
    if epoch == 60:
        alpha = 5e-3
    elif epoch == 100:
        alpha = 1e-3
    correct = 0
    count = 0
    for i in range(0, trainData.m, batch_size):
        # print i
        X, y = trainData.sample(batch_size, i)
        classifier.clearGradParam()
        y_pred = classifier.forward(X)
        # print y_pred
        loss = criterion.forward(y_pred, y)
        gradLoss = criterion.backward(y_pred, y)
        classifier.backward(gradLoss)

        for layer in classifier.layers:
            if layer.type == 'Linear':
                layer.W_decrement = momentum * layer.W_decrement + alpha * layer.gradW
                layer.B_decrement = momentum * layer.B_decrement + alpha * layer.gradB
                layer.W -= layer.W_decrement
                layer.B -= layer.B_decrement

        label = torch.argmax(y_pred, dim=1)
        correct += torch.sum(label == y - 1).item()
        count += len(y)

    print 'Epoch', epoch, 'complete'
    print 'Training Accuracy:', correct * 100 / count, '\b%'
    torch.save(classifier, 'Modelfile1')
