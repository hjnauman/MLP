import numpy as np

class Perceptron():
    def __init__(self, inputs : np.ndarray, weights : np.ndarray, bias, activation_threshold):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_threshold = activation_threshold

    def activation(self):
        # ReLU activation function
        z = np.dot(self.inputs, self.weights) + self.bias
        return 0 if z <= self.activation_threshold else z