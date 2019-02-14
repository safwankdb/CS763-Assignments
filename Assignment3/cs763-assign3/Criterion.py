import torch
from torch.autograd import variable


class Criterion:
    def forward(inp, target):
        new = torch.zeros_like(target)
        for i in range(target.size()[0]):
            new[i][0] = inp[i][target[i]]
        return -torch.mean(torch.log(new))

    def backward(input, target):
        new = torch.zeros_like(target)
        for i in range(target.size()[0]):
            new[i] = -1 / (inp[i][target[i]])
        grad = new / target.size()[0]
        new1 = torch.zeros_like(input)
        for i in range(target.size()[0]):
            new1[i][target[i]] = grad[i]
        return new1
