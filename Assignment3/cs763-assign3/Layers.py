import torch

class Linear():
    """docstring for Linear"""

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.W = torch.rand(n_input, n_output)
        self.B = torch.rand(1, n_output)
        self.gradW = torch.zeros(n_input, n_output)
        self.gradB = torch.zeros(1, n_output)


    def forward(input):
    	self.output = input @ self.W + self.B
    	return self.output

    def backwards(input, gradOutput)
    	self.gradW = input.t() * gradOutput # Correct
    	self.gradB = torch.sum(gradOutput, axis=0, keepdims=True) # Correct
    	self.gradInput = gradOutput @ W.t() # Correct
        return gradInput


class ReLU():

    def __init__(self):
        self.output
        self.gradInput

    def forward(input):
        self.output = input * (input > 0)
        return self.output

    def backward(input, gradOutput)
		self.gradInput = gradOutput * (1 * (input > 0))
		return self.gradInput