import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self._layers = layers
        self._params = [layer.get_parameters() for layer in self._layers]
        self._grads = [layer.get_gradients() for layer in self._layers]
     
    def forward_prop(self, input, is_train):
        for layer in self._layers:
            input = layer.forward_prop(input, is_train)
        return input
        
    def backward_prop(self, dA_cur_layer, reg_lambda):
        for layer in reversed(self._layers):
            dA_cur_layer = layer.backward_prop(dA_cur_layer, reg_lambda)
    
    def summary(self):
        return '\n'.join((['Layer {}: {}'.format(i + 1, layer.summary()) for i, layer in enumerate(self._layers)]))
    
    def get_layers(self):
        return self._layers
    
    def get_parameters(self):
        return self._params
    
    def get_gradients(self):
        return self._grads
        