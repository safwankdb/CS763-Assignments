from dataset import Data
import torch

model = raw_input('Model name with path: ')
testData = Data(test=True)
correct = 0
count = 0
batch_size = 256


for i in range(0, testData.m, batch_size):
    batch_score = 0
    X, y = testData.sample(batch_size, i)
    classifier = torch.load(model)
    y_pred = classifier.forward(X)
    # print X.shape, y.shape, y_pred.shape
    label = torch.argmax(y_pred, dim=1)
    # print label.shape
    batch_score = torch.sum(label == (y - 1)).item()
    correct += batch_score
    count += batch_size
    print 'Batch Accuracy:', batch_score * 100 / batch_size

print 'Test Accuracy:', correct * 100 / count
