from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def step():
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, nn_parameters, nn_gradients):
        for i in range(len(nn_parameters)):
            nn_parameters[i]['W'] = nn_parameters[i]['W'] - self.learning_rate * nn_gradients[i]['dW']
            nn_parameters[i]['b'] = nn_parameters[i]['b'] - self.learning_rate * nn_gradients[i]['db']