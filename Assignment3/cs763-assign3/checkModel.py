import argparse
import torchfile
import torch
from Model import Model
from Layers import Linear,ReLU


argument = argparse.ArgumentParser()
argument.add_argument("-config")
argument.add_argument("-i")
argument.add_argument("-og")
argument.add_argument("-o")
argument.add_argument("-ow")
argument.add_argument("-ob")
argument.add_argument("-ig")
parser = argument.parse_args()

myModel = Model()

#remove occurences of /n in all strings
with open(parser.config) as f:
    arr = f.readlines()
    # print("ARR",arr)
    num_layers = int(arr[0].replace('\n',''))
    # print("NUM",num_layers)
    for i in range(1, num_layers + 1):
        arr[i]=arr[i].replace('\n','')
        v = arr[i].split(' ')
        # print("V0",v[0])
        # print("LAST",v[0][-1])
        if(v[0] == "relu"):
            myModel.addLayer(ReLU())
        elif(v[0] == "linear"):
            # print("YO")
            inp = int(v[1])
            out = int(v[2])
            myModel.addLayer(Linear(inp, out))
    layer_weights_path = arr[num_layers + 1]
    layer_bias_path = arr[num_layers + 2]
    layer_bias_path=layer_bias_path.replace('\n','')
    layer_weights_path=layer_weights_path.replace('\n','')

weights = torchfile.load(layer_weights_path)
# weights = torch.from_numpy(weights)
bias = torchfile.load(layer_bias_path)
# bias = torch.from_numpy(bias)

index = 0

for layer in myModel.layers:
    if(type(layer) == Linear):
        layer.W = torch.from_numpy(weights[index]).float()
        # print(layer.W.size())
        # print(layer.W)
        layer.B = torch.from_numpy(bias[index]).view(-1,1).float()
        # print(layer.B.size())
        # print(layer.B)
        index += 1

num_linear = index
input = torchfile.load(parser.i)
input = torch.from_numpy(input)
input= input.float()
batch_size=input.shape[0]
# print(input)
input=input.view(batch_size,-1)
# print(input)
gradOutput = torchfile.load(parser.og)
gradOutput = torch.from_numpy(gradOutput)
gradOutput=gradOutput.float()

output=myModel.forward(input)
print(output)
torch.save(output,parser.o)

# gradInput=myModel.backward(gradOutput)
# gradW=[]
# gradB=[]
# for layer in myModel.layers:
#     if(type(layer) == type(Linear)):
#         gradW.append(layer.gradW)
#         gradB.append(layer.gradB)
# gradB=torch.tensor(gradB)
# gradW=torch.tensor(gradW)
# torch.save(gradB,parser.ob)
# torch.save(gradW,parser.ow)
# torch.save(gradInput, parser.ig)