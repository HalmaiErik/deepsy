import numpy as np

from deepsy.core.functions.activation_functions import ActivationFunction, ReLU


class Layer:
    def __init__(self, nr_neurons, nr_in_features, activation_func: ActivationFunction):
        self._nr_neurons = nr_neurons
        self._nr_in_features = nr_in_features
        self._activation_func = activation_func
        self._init_parameters_and_gradients()

    def forward_prop(self, X):
        assert self._params['W'].shape[1] == X.shape[0], 'Shape of input = {} does not correspond to the shape of the weights = {}'.format(X.shape, self._params['W'].shape)
        self.X = X
        self.Z = np.dot(self._params['W'], self.X) + self._params['b']
        return self._activation_func.activate(self.Z)
    
    def backward_prop(self, dA):
        dZ = dA * self._activation_func.derivate(self.Z)
        self._grads['dW'] = np.dot(dZ, self.X.T) / dZ.shape[1]
        self._grads['db'] = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]
        return np.dot(self._params['W'].T, dZ)
    
    def summary(self):
        return 'Activation function={}, # Input features={}, # Neurons={}, Weight shape={}, Bias shape={}, # Parameters={}'\
                    .format(self._activation_func.__class__.__name__, self._nr_in_features, self._nr_neurons, self._params['W'].shape, self._params['b'].shape, self._nr_parameters)
    
    def get_parameters(self):
        return self._params
    
    def get_gradients(self):
        return self._grads
    
    def _init_parameters_and_gradients(self):
        W = np.random.randn(self._nr_neurons, self._nr_in_features)
        if (isinstance(self._activation_func, ReLU)):
            # He init
            W *= np.sqrt(2. / self._nr_in_features)
        else:
            # Xavier init
            W *= np.sqrt(1. / self._nr_in_features)

        b = np.zeros((self._nr_neurons, 1))

        self._params = {'W': W,
                       'b': b}
        self._grads = {'dW': np.zeros(self._params['W'].shape),
                      'db': np.zeros(self._params['b'].shape)}
        self._nr_parameters = self._params['W'].shape[0] * self._params['W'].shape[1] + self._params['b'].shape[0] * self._params['b'].shape[1]

