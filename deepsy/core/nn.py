import numpy as np
from deepsy.core.functions.loss_functions import get_loss_function


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward_prop(self, input):
        for layer in self.layers:
            input = layer.forward_prop(input)
        
    def backward_prop(self, losses):
        dA_cur_layer = self.loss.derivate(losses)
        for layer in reversed(self.layers):
            dA_cur_layer = layer.backward_prop(dA_cur_layer)
    
    def summary(self):
        return 'Model summary:\nLoss={}\n'.format(self.loss_name) + ''.join((['Layer {}: {}'.format(i + 1, layer.summary()) for i, layer in enumerate(self.layers)]))
    
    def get_layer(self):
        return self.layers
        