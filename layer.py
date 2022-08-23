from turtle import forward


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # Implement forward propagation
        # TODO: return output
        pass

    # We can additionally add an optimizer here. Need to implement later
    def backward(self, output_gradient, learning_rate):
        # Implement backward propagation
        # TODO: update parameters and return input gradient
        pass
