import argparse
import torchfile
import Model

argument = argparse.ArgumentParser()
argument.add_argument("--config")
argument.add_argument("--i")
argument.add_argument("--og")
argument.add_argument("--o")
argument.add_argument("--ow")
argument.add_argument("--ob")
argument.add_argument("--ig")
parser = argument.parse_args()

myModel=Model()

with open(parser.config) as f:
	arr=f.readlines()
	int num_layers=int(arr[0])
	for i in range(1,num_layers+1):
		v=arr[0].split(' ')
		if(v[0] is "relu"):
			myModel.addLayer(ReLU())
		else if(v[0] is "linear"):
			inp=int(v[1])
			out=int(v[2])
			myModel.addLayer(Linear(inp,out))
	layer_weights_path=arr[num_layers+1]
	layer_bias_path=arr[num_layers+2]

weights = torchfile.load('layer_weights_path')
weights=torch.from_numpy(weights)
bias= torchfile.load('layer_bias_path')
bias=torch.from_numpy(bias)

index=0

for layer in myModel.layers:
	if(type(layer)==type(Linear)):
		layer.W=weights[index]
		layer.B=bias[index]
		index+=1

num_linear=index
input=torchfile.load(parser.i)
input=torch.from_numpy(input)
gradOutput=torchfile.load(parser.og)
gradOutput=torch.from_numpy(gradOutput)

