import argparse
import torch
from Criterion import Criterion
import torchfile

argument = argparse.ArgumentParser()
argument.add_argument("--i")
argument.add_argument("--t")
argument.add_argument("--ig")
parser = argument.parse_args()

inp = torchfile.load(parser.i)
inp = torch.from_numpy(inp).double()
target = torchfile.load(parser.t)
target = torch.from_numpy(target).long()

print('Average loss for given input and target is', Criterion.forward(inp, target))

gradients = Criterion.backward(inp, target)
gradients = gradients.numpy()
gradients.tofile(parser.ig)
