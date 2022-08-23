from activationLayer import Activition
import numpy as np

class ReLU(Activition):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime)