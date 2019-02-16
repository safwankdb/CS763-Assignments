import torch


class Criterion:
    def forward(inp, target):
        exponential = torch.exp(inp)
        softmax = exponential / (torch.sum(exponential, 1, True))
        new = torch.zeros_like(target)
        for i in range(target.size()[0]):
            new[i][0] = softmax[i][target[i]]
        return -torch.mean(torch.log(new))

    def backward(inp, target):
        exponential = torch.exp(inp)
        softmax = exponential / (torch.sum(exponential, 1, True))
        bsize = inp.size(0)
        softmax[range(bsize), target] -= 1
        grad = softmax / bsize
        return grad
