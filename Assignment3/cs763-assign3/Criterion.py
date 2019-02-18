import torch


class Criterion:
    def forward(inp, target):
        bsize = inp.size()[0]
        exponential = torch.exp(inp).float()
        softmax = exponential / (torch.sum(exponential, 1, True))
        # one = torch.ones_like(target)
        # target = target - one
        log_likelihood = -torch.log(softmax[range(bsize), target - 1])
        loss = torch.sum(log_likelihood) / bsize
        return loss
        # new = torch.zeros_like(target).float()
        # for i in range(target.size()[0]):
        # new[i][0] = softmax[i][target[i]]
        # return torch.sum(log_likelihood)/

    def backward(inp, target):
        # one = torch.ones_like(target)
        # target = target - one
        exponential = torch.exp(inp).float()
        softmax = exponential / (torch.sum(exponential, 1, True))
        bsize = inp.size()[0]
        softmax[range(bsize), target - 1] -= 1
        grad = softmax / bsize
        return grad

        # new = torch.zeros_like(target)
        # for i in range(target.size()[0]):
        # new[i] = -1 / (inp[i][target[i]])
        # grad = new / target.size()[0]
        # new1 = torch.zeros_like(input)
        # for i in range(target.size()[0]):
        # new1[i][target[i]] = grad[i]
        # return new1
