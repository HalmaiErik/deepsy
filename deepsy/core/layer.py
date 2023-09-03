import numpy as np

from deepsy.core.functions.activation_functions import ActivationFunction, ReLU


class Layer:
    def __init__(self, nr_in_features, nr_neurons, activation_func: ActivationFunction):
        self.nr_in_features = nr_in_features
        self.nr_neurons = nr_neurons
        self.activation_func = activation_func
        self._init_parameters_and_gradients()

    def forward_prop(self, X):
        assert self.params['W'].shape[1] == X.shape[0], 'Shape of input = {} does not correspond to the shape of the weights = {}'.format(X.shape, self.params['W'].shape)
        self.X = X
        self.Z = np.dot(self.params['W'], self.X) + self.params['b']
        return self.activation_func.activate(self.Z)
    
    def backward_prop(self, dA):
        dZ = dA * self.activation_func.derivate(self.Z)
        self.grads['dW'] = np.dot(dZ, self.X.T) / dZ.shape[1]
        self.grads['db'] = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]
        return np.dot(self.params['W'].T, dZ)
    
    def summary(self):
        nr_parameters = self.params['W'].shape[0] * self.params['W'].shape[1] + self.params['b'].shape[0] * self.params['b'].shape[1]
        return 'Activation={}, # Input features={}, # Neurons={}, Weight shape={}, Bias shape={}, # Parameters={}\n'\
                    .format(self.activation_func.__class__.__name__, self.nr_in_features, self.nr_neurons, self.params['W'].shape, self.params['b'].shape, nr_parameters)
    
    def get_parameters(self):
        return self.params['W'], self.params['b']
    
    def _init_parameters_and_gradients(self):
        W = np.random.randn(self.nr_neurons, self.nr_in_features)
        if (isinstance(self.activation_name, ReLU)):
            # He init
            W *= np.sqrt(2. / self.nr_in_features)
        else:
            # Xavier init
            W *= np.sqrt(1. / self.nr_in_features)

        b = np.zeros((self.nr_neurons, 1))

        self.params = {'W': W,
                       'b': b}
        self.grads = {'dW': np.zeros(self.params['W'].shape),
                      'db': np.zeros(self.params['b'].shape)}

