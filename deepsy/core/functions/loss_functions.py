from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def get_loss(self, Y_predicted, Y_actual):
        pass


class MSE(LossFunction):
    def get_loss(self, Y_predicted, Y_actual):
        return 0.5 * np.square(Y_predicted - Y_actual)
    

class BinaryCrossEntropy(LossFunction):
    def get_loss(self, Y_predicted, Y_actual):
        return -Y_actual * np.log(Y_predicted) - (1 - Y_actual) * np.log(1 - Y_predicted)


class SoftmaxCrossEntropy(LossFunction):
    def get_loss(self, Y_predicted, Y_actual):
        return -np.sum(Y_actual * np.log(Y_predicted), axis=0, keepdims=True)