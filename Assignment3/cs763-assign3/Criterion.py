import torch


class Criterion:
    def __init__(self):
        return

    def forward(self, inp, target):
        exponential = torch.exp(inp).double()
        softmax = exponential / (torch.sum(exponential, 1, True))
        target = target - 1
        new = torch.zeros_like(target).double()
        for i in range(target.size()[0]):
            new[i] = softmax[i][target[i]]
        return -torch.mean(torch.log(new))

    def backward(self, inp, target):
        target = target - 1
        exponential = torch.exp(inp).double()
        softmax = exponential / (torch.sum(exponential, 1, True))
        bsize = inp.size()[0]
        for i in range(bsize):
            softmax[i][target[i]] = softmax[i][target[i]] - 1
        grad = softmax / bsize
        return grad
