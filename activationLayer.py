from turtle import forward
from layer import Layer
import numpy as np


class Activition(Layer):
    # Activation prime is the derivative of activation
    # Both activation and activation_prime are funtions
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
