import numpy as np
from deepsy.core.functions.loss_functions import LossFunction
from deepsy.core.functions.optimizers import Optimizer
from deepsy.core.nn import NeuralNetwork


class Model:
    def __init__(self, neural_network: NeuralNetwork, loss_func: LossFunction, optimizer: Optimizer):
        self._neural_network = neural_network
        self._loss_func = loss_func
        self.optimizer = optimizer

    def train(self, X, Y, nr_epochs, reg_lambda=0.0):
        for epoch in range(nr_epochs):
            # forward propagation
            Y_predicted = self._neural_network.forward_prop(input=X, is_train=True)

            # compute cost
            cost = np.mean(self._loss_func.get_loss(Y_predicted, Y))
            if reg_lambda > 0:
                cost += self._calc_reg_cost_term(reg_lambda, X.shape[1])

            # backward propagation: compute gradients for each layer
            self._neural_network.backward_prop(Y_predicted - Y, reg_lambda)

            # update gradients
            self.optimizer.step(nn_parameters=self._neural_network.get_parameters(), nn_gradients=self._neural_network.get_gradients())

            print('Epoch {}: cost = {}'.format(epoch + 1, cost))

    def validate(self, X, Y):
        Y_predicted = self._neural_network.forward_prop(input=X, is_train=False)
        cost = np.mean(self._loss_func.get_loss(Y_predicted, Y))
        return Y_predicted, cost

    def predict(self, X):
        return self._neural_network.forward_prop(X, is_train=False)
    
    def summary(self):
        print('Model summary:')
        print('Loss function={}'.format(self._loss_func.__class__.__name__))
        print(self._neural_network.summary())
    
    def get_neural_network(self):
        return self._neural_network
    
    def _calc_reg_cost_term(self, reg_lambda, m):
        reg_term = 0.0
        for layer_params in self._neural_network.get_parameters():
            reg_term += np.sum(np.square(layer_params['W']))
        reg_term *= 0.5 * reg_lambda / m
        return reg_term