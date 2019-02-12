import torch


class Linear():
    """A Fully Connected Linear Layer"""

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.W = torch.rand(n_input, n_output)
        self.B = torch.rand(1, n_output)
        self.gradW = torch.zeros(n_input, n_output)
        self.gradB = torch.zeros(1, n_output)

    def forward(self, input):
        """Forward Pass"""
        self.output = input @ self.W + self.B
        return self.output

    def backwards(self, input, gradOutput):
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
