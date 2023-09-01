from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, Z):
        pass


class Linear(ActivationFunction):
    def activate(self, Z):
        return Z
    

class Sigmoid(ActivationFunction):
    def activate(self, Z):
        return 1 / (1 + np.exp(-Z))
    

class Tanh(ActivationFunction):
    def activate(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))    
    

class ReLU(ActivationFunction):
    def activate(self, Z):
        return np.maximum(0, Z)
    

class LeakyReLU(ActivationFunction):
    def activate(self, Z):
        return np.maximum(0.01 * Z, Z)
    

_activations = {
        'linear': Linear,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'leaky-relu': LeakyReLU
}

def get_activation_function(activation_name):
    activation = _activations.get(activation_name)
    if activation is None:
        raise Exception('{} not a known activation function name'.format(activation_name))
    return activation()
