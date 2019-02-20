import torch
from math import sqrt

device = 'cuda:0'


class Linear():
    """
    A Fully Connected Linear Layer. - Mohd Safwan

    Dimensions:
    input , gradInput  -> batch_size x n_input
    output, gradOutput -> batch_size x n_output
    W     , gradW      -> n_output   x n_input
    B     , gradB      -> n_output   x 1

    """

    def __init__(self, n_input, n_output):
        """Layer Initialization"""
        self.type = 'Linear'
        self.W_decrement = torch.zeros(n_output, n_input).double().to(device)
        self.B_decrement = torch.zeros(n_output, 1).double().to(device)
        self.W = torch.randn(n_output, n_input).double().to(device) / sqrt(n_input)
        self.B = torch.randn(n_output, 1).double().to(device) / sqrt(n_input)
        # earlier it was 1D vector -> wrong
        self.gradW = torch.zeros_like(self.W)
        self.gradB = torch.zeros_like(self.B)

    def forward(self, inp):
        """Forward Pass"""
        self.batch_size = inp.shape[0]
        self.pad = torch.ones(self.batch_size, 1).double().to(device)
        # print(self.pad.dtype)
        self.output = torch.matmul(inp, self.W.t().to(device)) \
            + torch.matmul(self.pad, self.B.t().to(device))
        # print self.output
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        self.gradW = torch.matmul(gradOutput.t(), input)
        self.gradB = torch.sum(gradOutput, dim=0).view(-1, 1)
        self.gradInput = torch.matmul(gradOutput, self.W)
        return self.gradInput

    def clear_grad(self):
        self.gradW = torch.zeros_like(self.gradW)
        self.gradB = torch.zeros_like(self.gradB)

    def print_params(self):
        r = self.W.shape[0]
        c = self.W.shape[1]
        s = ''
        for i in range(r):
            for j in range(c):
                s += str(self.W[i][j].item()) + ' '
            s += str(self.B[i].item()) + '\n'
        print s


class ReLU():
    """A Residual Linear Unit Activation Layer """

    def __init__(self):
        """Layer Initialization"""
        self.type = 'ReLU'

    def forward(self, input):
        """Forward Pass"""
        self.output = input * (input > 0).double()
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        self.gradInput = gradOutput * (1.0 * (input > 0).double())
        return self.gradInput

    def clear_grad(self):
        pass

    def print_params(self):
        pass


class Convolution():
    """INCOMPLETE Convolutional Layer"""

    """
    TODO
    - Decide Initialization parameters
    - Implement multiple kernels
    - Implement Backpropagation
    """

    def __init__(self, n_input, n_output, kernel_size, n_kernels):
        """
        Layer Initialization

        kernel_size is a tuple-like object
        """
        self.kernel = torch.rand(kernel_size)

    def convolve(image, kernels):
        """Convolve an Image with a Kernel"""
        assert image.shape[-1] == kernels.shape[-1]
        res_len = len(image) - len(kernels[0]) + 1
        depth = len(kernels)
        result = torch.zeros(res_len, res_len, depth)
        for row in range(res_len):
            for col in range(res_len):
                for k_i in range(depth):
                    kernel = kernels[k_i]
                    patch = image[col:col + len(kernel), row:row + len(kernel)]
                    filtered = patch * kernel
                    result[row, col, k_i] = torch.sum(filtered)
        return result
