from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    def __init__(self):
        self.Y_predicted = np.zeros((1, 1))
        self.Y = np.zeros((1, 1))

    @abstractmethod
    def get_loss(self, Y_predicted, Y):
        pass

    @abstractmethod
    def derivate(self):
        pass


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return 0.5 * np.square(self.Y_predicted - self.Y)
    
    def derivate(self):
        return self.Y_predicted - self.Y
    

class BinaryCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return -self.Y * np.log(self.Y_predicted) - (1 - self.Y) * np.log(1 - self.Y_predicted)
    
    def derivate(self):
        return -self.Y / self.Y_predicted + (1 - self.Y) / (1 - self.Y_predicted)


class SoftmaxCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return -np.sum(self.Y * np.log(self.Y_predicted), axis=0, keepdims=True)
    
    def derivate(self):
        return self.Y_predicted - self.Y
    