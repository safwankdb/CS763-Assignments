import torch
from torch.autograd import variable


class Criterion:
    def forward(inp, target):
        out = input * torch.diag(target)
        return -torch.mean(torch.log(out))

    def backward(input, target):
        
