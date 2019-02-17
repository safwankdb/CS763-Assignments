import torch


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
        self.W = torch.rand(n_output, n_input)
        self.B = torch.rand(n_output, 1) #earlier it was 1D vector -> wrong
        self.gradW = torch.zeros(n_output, n_input)
        self.gradB = torch.zeros(n_output, 1)

    def forward(self, Input):
        """Forward Pass"""
        self.batch_size = Input.shape[0]
        self.pad = torch.ones(self.batch_size, 1)
        # print(Input.size()) 
        # print(self.W.t().size())
        # print(self.pad.size())
        # print(self.B.t().size())
        self.output = Input @ self.W.t() + self.pad @ self.B.view(-1,1).t()
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        # print(input.size())
        # print(gradOutput.t().size())
        self.gradW = gradOutput.t() @ input
        self.gradB = torch.sum(gradOutput, dim=0)
        # print(self.gradB.size())
        # print(self.B.size())
        self.gradInput = gradOutput @ self.W
        return self.gradInput

    def clear_grad(self):
        self.W = torch.zeros(self.W.shape)
        self.B = torch.zeros(self.B.shape)

    def print_params(self):
        r = self.W.shape[0]
        c = self.W.shape[1]
        for i in range(r):
            for j in range(c):
                print(self.W[i][j])
            print


class ReLU():
    """A Residual Linear Unit Activation Layer """

    def __init__(self):
        """Layer Initialization"""

    def forward(self, input):
        """Forward Pass"""
        self.output = input * (input > 0).float()
        return self.output

    def backward(self, input, gradOutput):
        """Backward Pass"""
        self.gradInput = gradOutput * (1.0 * (input > 0).float())
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
