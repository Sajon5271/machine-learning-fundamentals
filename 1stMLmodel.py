from msilib.schema import File
from re import A
import numpy as np
import gzip
import idx2numpy

import matplotlib.pyplot as plt


def flattenImageData(data):
    processed = []
    for image in data:
        processed.append(image.flatten())

    processed = np.asarray(processed)
    return processed


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(X):
    return np.maximum(0, X)


def deriv_relu(X):
    return X > 0


def softmax(X):
    A = np.exp(X) / sum(np.exp(X))
    return A


def one_hot(Y):
    one_hot_y = np.zeros((Y.size, 10))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    # 1st Activation Hidden Layer
    A1 = ReLU(Z1)
    # print(A1.shape)
    Z2 = W2.dot(A1) + b2
    # This is the Output
    Y = softmax(Z2)
    return Z1, A1, Z2, Y


def backward_propagation(Z1, A1, Z2, Y, W2, X):
    n = train_labels_raw.size
    # Finding error in our output
    dZ2 = Y - train_labels
    # Dependency on W2
    dW2 = 1 / n * dZ2.dot(A1.T)
    # db2 = 1 / n * np.sum(dZ2, 2) *******Check if this does anything
    db2 = 1 / n * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / n * dZ2.dot(X.T)
    # db2 = 1 / n * np.sum(dZ2, 2) *******Check if this does anything
    db1 = 1 / n * np.sum(dZ1)
    return dW1, db1, dW2, db2


# def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
#     W1 = W1 - alpha * dW1
#     b1 = b1 - alpha * db1
#     # print(W2.shape, dW2.shape)
#     W2 = W2 - alpha * dW2
#     b2 = b2 - alpha * db2
#     return W1, b1, W2, b2


# def get_predictions(A):
#     return np.argmax(A, 0)


# def get_accuracy(Predictions, Y):
#     # print(Predictions, Y)
#     return np.sum(Predictions == Y) / Y.size


# def gradient_descent(X, Y, iterations, alpha):
#     W1, b1, W2, b2 = init_params()
#     for i in range(iterations):
#         Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
#         dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)
#         W1, b1, W2, b2 = update_params(
#             W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if i % 10 == 0:
#             print("Iteration: ", i)
#             print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

#     return W1, b1, W2, b2


file1 = gzip.open('train-images-idx3-ubyte.gz')
file2 = gzip.open('train-labels-idx1-ubyte.gz')
file3 = gzip.open('t10k-images-idx3-ubyte.gz')
file4 = gzip.open('t10k-labels-idx1-ubyte.gz')
train_image_raw = idx2numpy.convert_from_file(file1)
train_labels_raw = idx2numpy.convert_from_file(file2)
test_image_raw = idx2numpy.convert_from_file(file3)
test_labels_raw = idx2numpy.convert_from_file(file4)

count, shape_x, shape_y = train_image_raw.shape
train_image = flattenImageData(train_image_raw)
train_image = train_image.T
train_labels = one_hot(train_labels_raw)
print(train_labels.shape)
print(train_image.shape)

# W1, b1, W2, b2 = gradient_descent(train_image, train_labels_raw, 100, 0.01)
# print(train_labels_raw.size)
# # Visualize Data as Image
# # plt.imshow(train_image[2,:,:], cmap='gray')
# # print(type(train_image))
# # plt.show()
