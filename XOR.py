import imp
from sys import implementation
from denseLayer import Dense
from ReLU import ReLU
from mse import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    ReLU(),
    Dense(3, 1),
    ReLU()
]

epochs = 10000
learning_rate = 0.1

# Train
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # Error
        error += mse(y, output)

        # Backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (e+1, epochs, error))
