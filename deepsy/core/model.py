import numpy as np
from deepsy.core.functions.loss_functions import LossFunction
from deepsy.core.nn import NeuralNetwork


class Model:
    def __init__(self, neural_network: NeuralNetwork, loss_func: LossFunction):
        self.neural_network = neural_network
        self.loss_func = loss_func

    def train(self, X, Y, nr_epochs):
        for epoch in nr_epochs:
            # forward propagation
            Y_predicted = self.neural_network.forward_prop(input=X)

            # compute loss & cost
            L = self.loss_func.get_loss(Y_predicted, Y)
            cost = np.mean(L)

            # backward propagation: compute gradients for each layer
            self.neural_network.backward_prop(L)

            # TODO: update gradients

    def validate(self, X, Y):
        Y_predicted = self.neural_network.forward_prop(input=X)
        cost = np.mean(self.loss_func.get_loss(Y_predicted, Y))
        return Y_predicted, cost

    def predict(self, X):
        return self.neural_network.forward_prop(X)
    
    def summary(self):
        print('Model summary:')
        print('Loss function={}'.format(self.loss_func.__class__.__name__))
        print(self.neural_network.summary())
    
    def get_neural_network(self):
        return self.neural_network