import torch

'''
TODO
Treat input as batch
'''


class Linear():
    """A Fully Connected Linear Layer"""

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.W = torch.rand(n_output, n_input)
        self.B = torch.rand(n_output, 1)
        self.gradW = torch.zeros(n_output, n_input)
        self.gradB = torch.zeros(n_output, 1)
    """
    Dimensions:

    input, gradInput   -> batch_size x n_input
    output, gradOutput -> batch_size x n_output
    W, gradW           -> n_output x n_input
    B, gradB           -> n_output x 1

    """

    def forward(self, input):
        """Forward Pass"""
        self.pad = torch.ones(self.batch_size, 1)
        self.batch_size = input.shape[0]
        self.output = input @ self.W.t() + self.pad @ self.B.t()
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        self.gradW = input.t() @ gradOutput
        self.gradB = torch.sum(gradOutput, axis=0, keepdims=True)
        self.gradInput = gradOutput @ self.W.t()
        return self.gradInput


class ReLU():
    """A Residual Linear Unit Activation Layer """

    def __init__(self):
        self.output
        self.gradInput

    def forward(self, input):
        """Forward Pass"""
        self.output = input * (input > 0)
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        self.gradInput = gradOutput * (1 * (input > 0))
        return self.gradInput
