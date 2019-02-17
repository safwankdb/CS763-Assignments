import torchfile
import torch

out2= torchfile.load("files/output_sample_1.bin")
out2= torch.from_numpy(out2)
print(out2.size())
print(out2)

print("OTHER")

out1 = torchfile.load("myout_1.bin")
out1 = torch.from_numpy(out1)
print(input)
