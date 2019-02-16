class Model():

    def __init__(self):
        '''
        layers are to be stored according to the order of forward propagation
        '''
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        cur = input
        self.input = cur
        for layer in self.layers:
            cur = layer.forward(cur)
        return cur

    def clearGradParam(self):
        for layer in self.layers:
            layer.clear_grad()

    def dispGradParam(self):
        '''
        only prints W matrix parameter of Linear layer, not Bias
        '''
        for layer in reversed(self.layers):
            layer.print_params()

    def backward(self, gradOutput):
        cur = gradOutput
        revLayers = reversed(self.layers)
        for i in range(len(revLayers) - 1):
            cur = revLayers[i].backward(revLayers[i + 1].output, cur)
        if(len(revLayers) > 0):
            revLayers[len(revLayers) - 1].backward(self.input, cur)
