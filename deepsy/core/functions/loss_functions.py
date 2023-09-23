from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    def __init__(self):
        self.Y_predicted = np.zeros((1, 1))
        self.Y = np.zeros((1, 1))

    @abstractmethod
    def get_loss(self, Y_predicted, Y):
        pass


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return 0.5 * (self.Y_predicted - self.Y) ** 2
    

class BinaryCrossEntropy(LossFunction):
    EPSILON = 1e-5

    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return -self.Y * np.log(self.Y_predicted + self.EPSILON) - (1 - self.Y) * np.log(1 - self.Y_predicted + self.EPSILON)


class SoftmaxCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, Y_predicted, Y):
        self.Y_predicted = Y_predicted
        self.Y = Y
        return -np.sum(self.Y * np.log(self.Y_predicted), axis=0, keepdims=True)
