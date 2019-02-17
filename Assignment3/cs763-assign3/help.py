"""
Issues left:- 
1) Use double(float64) instead of float
2) Saving properly to file
3) elements of list in output of gradB and gradW etc. should be numpy arrays not tensors
"""

import torchfile
import torch

out2= torchfile.load("files/gradW_sample_2.bin")
print(out2)

# print("OTHER")

# out1 = torchfile.load("myout_1.bin")
# out1 = torch.from_numpy(out1)
# print(input)
