class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward_prop(self, input, expected_output=None):
        for layer in self.layers:
            input = layer.forward_prop(input)
        return input
    
    def summary(self):
        return 'Model summary:\n' + ''.join((['Layer {}: {}'.format(i + 1, layer.summary()) for i, layer in enumerate(self.layers)]))
    
    def get_layer(self):
        return self.layers
        