import argparse
import Criterion
import torchfile

argument = arparse.ArgumentParser()
argument.add_argument("--i")
argument.add_argument("--t")
argument.add_argument("--ig")
parser = argument.parse_args()

myCriterion = Criterion()

inp = torchfile.load(parser.i)
inp = torch.from_numpy(inp)
target = torchfile.load(parser.t)
target = torch.from_numpy(target)

print('Average loss for given input and target is', Criterion.forward(inp, target))

gradients = Criterion.backward(inp, target)
gradients = gradients.numpy()
gradints.tofile(parser.ig)
