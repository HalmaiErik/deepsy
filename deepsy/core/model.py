import numpy as np
from deepsy.core.functions.loss_functions import LossFunction
from deepsy.core.functions.optimizers import Optimizer
from deepsy.core.nn import NeuralNetwork


class Model:
    def __init__(self, neural_network: NeuralNetwork, loss_func: LossFunction, optimizer: Optimizer):
        self._neural_network = neural_network
        self._loss_func = loss_func
        self.optimizer = optimizer

    def train(self, X, Y, nr_epochs):
        for epoch in range(nr_epochs):
            # forward propagation
            Y_predicted = self._neural_network.forward_prop(input=X)

            # compute loss & cost
            L = self._loss_func.get_loss(Y_predicted, Y)
            cost = np.mean(L)

            # backward propagation: compute gradients for each layer
            self._neural_network.backward_prop(self._loss_func.derivate())

            # update gradients
            self.optimizer.step(nn_parameters=self._neural_network.get_parameters(), nn_gradients=self._neural_network.get_gradients())

            print('Epoch {}: cost = {}'.format(epoch + 1, cost))

    def validate(self, X, Y):
        Y_predicted = self._neural_network.forward_prop(input=X)
        cost = np.mean(self._loss_func.get_loss(Y_predicted, Y))
        return Y_predicted, cost

    def predict(self, X):
        return self._neural_network.forward_prop(X)
    
    def summary(self):
        print('Model summary:')
        print('Loss function={}'.format(self._loss_func.__class__.__name__))
        print(self._neural_network.summary())
    
    def get_neural_network(self):
        return self._neural_network