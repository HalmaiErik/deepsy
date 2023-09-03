from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        self.G = np.zeros((1, 1))

    @abstractmethod
    def activate(self, Z):
        pass
    
    @abstractmethod
    def derivate(self, Z):
        pass

class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return Z
    
    def derivate(self, Z):
        return np.ones_like(Z)

class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        self.G = 1 / (1 + np.exp(-Z))
        return self.G
    
    def derivate(self, Z):
        return self.G * (1 - self.G)


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        self.G = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))   
        return self.G
    
    def derivate(self, Z):
        return 1 - self.G ** 2
    

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0, Z)
    
    def derivate(self, Z):
        return (Z >= 0).astype(Z.dtype)


class LeakyReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0.01 * Z, Z)
    
    def derivate(self, Z):
        return np.where(Z >=0, 1., 0.01)
    

class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()
    
    def activate(self, Z):
        exp_Z = np.exp(Z)
        sum = np.sum(exp_Z, axis=1, keepdims=True)
        self.G = exp_Z / sum
        return self.G
    
    def derivate(self, Z):
        return self.G