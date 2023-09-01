from deepsy.core.nn import NeuralNetwork


class Model:
    def __init__(self, neural_network: NeuralNetwork):
        self.neural_network = neural_network

    def predict(self, X):
        return self.neural_network.forward_prop(X)
    
    def summary(self):
        print(self.neural_network.summary())
    
    def get_neural_network(self):
        return self.neural_network