class Model():

    def __init__(self, isTrain):
        '''
        layers are to be stored according to the order of forward propagation
		isTrain is true when the model is training and false otherwise
        '''
        self.isTrain=isTrain
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        cur = input
        self.input = cur
        for layer in self.layers:
            cur = layer.forward(cur,isTrain)
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
        revLayers = list(reversed(self.layers))
        for i in range(len(revLayers) - 1):
            cur = revLayers[i].backward(revLayers[i + 1].output, cur)
        if(len(revLayers) > 0):
            cur = revLayers[len(revLayers) - 1].backward(self.input, cur)
        return cur
