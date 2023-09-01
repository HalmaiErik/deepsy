import numpy as np

from deepsy.core.functions.activation_functions import get_activation_function


class Layer:
    def __init__(self, nr_in_features, nr_neurons, activation='linear'):
        self.nr_in_features = nr_in_features
        self.nr_neurons = nr_neurons
        self.activation_name = activation
        self.activation = get_activation_function(self.activation_name)
        self._init_parameters()

    def forward_prop(self, X):
        return self._compute_activation_values(X)
    
    def summary(self):
        nr_parameters = self.weights.shape[0] * self.weights.shape[1] + self.biases.shape[0] * self.biases.shape[1]
        return 'Activation={}, # Input features={}, # Neurons={}, Weight shape={}, Bias shape={}, # Parameters={}\n'\
                    .format(self.activation_name, self.nr_in_features, self.nr_neurons, self.weights.shape, self.biases.shape, nr_parameters)
    
    def get_parameters(self):
        return self.weights, self.biases
    
    def _init_parameters(self):
        self.weights = np.random.randn(self.nr_neurons, self.nr_in_features)
        if (self.activation_name == 'relu'):
            # He init
            self.weights *= np.sqrt(2. / self.nr_in_features)
        else:
            # Xavier init
            self.weights *= np.sqrt(1. / self.nr_in_features)

        self.biases = np.zeros((self.nr_neurons, 1))

    def _compute_activation_values(self, X):
        assert self.weights.shape[1] == X.shape[0], 'Shape of input = {} does not correspond to the shape of the weights = {}'.format(X.shape, self.weights.shape)
        return self.activation.activate(np.dot(self.weights, X) + self.biases)