import numpy as np
from deepsy.core.functions.loss_functions import LossFunction
from deepsy.core.functions.optimizers import Optimizer
from deepsy.core.nn import NeuralNetwork


class Model:
    def __init__(self, neural_network: NeuralNetwork, loss_func: LossFunction, optimizer: Optimizer):
        self._neural_network = neural_network
        self._loss_func = loss_func
        self.optimizer = optimizer

    def train(self, X, Y, nr_epochs, batch_size=64):
        for epoch in range(nr_epochs):
            batch_start = 0
            cost = 0
            nr_batches = float(len(X) / batch_size)
            while (batch_start < len(X)):
                batch_end = min(batch_start + batch_size, len(X[0]))
                X_batch = X[:, batch_start : batch_end]
                Y_batch = Y[batch_start : batch_end]

                # forward propagation
                Y_batch_predicted = self._neural_network.forward_prop(input=X_batch)

                # compute loss & cost
                cost += np.mean(self._loss_func.get_loss(Y_batch_predicted, Y_batch))

                # backward propagation: compute gradients for each layer
                self._neural_network.backward_prop(self._loss_func.derivate())

                # update gradients
                self.optimizer.step(nn_parameters=self._neural_network.get_parameters(), nn_gradients=self._neural_network.get_gradients())

                batch_start += batch_size

            cost = cost / nr_batches
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