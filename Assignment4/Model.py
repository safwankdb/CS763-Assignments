import torch
import numpy as np
from math import sqrt

class Model():

	def __init__(self, layer, numLayers=1):
        '''
        numLayers=1
        '''
        self.numLayers=numLayers
        self.layer=layer

    def forward(self, input):
    	return self.layer.forward(input)

    def backward(self, gradOut):
    	self.layer.backward(gradOut)

    def reset(self):
    	self.layer.reset()

        