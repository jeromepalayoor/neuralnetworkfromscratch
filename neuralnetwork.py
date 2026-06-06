import numpy as np

# hidden layer
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # initialise weights and biases
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        # forward propagation
        Z = np.dot(X, self.W) + self.b
        self.X = X
        return Z
    
    def backward(self, dZ):
        # backward propagation
        # get the loss gradient and use chain rule to find the parameter gradients
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        self.dW = dW
        self.db = db
        return dX

# rectified linear unit activation layer
class ReLULayer:
    def forward(self, Z):
        # relu func is 0 if x <= 0 or x if x > 0
        self.Z = Z
        return np.maximum(Z, 0)
    
    def backward(self, dA):
        # the derivative of relu 0 if x <= 0 or 1 if x > 0
        return dA * (self.Z > 0)

# softmax activation layer
class SoftmaxLayer:
    def forward(self, Z):
        # softmax is usually output layer
        # shift z to keep within bounds
        shift_Z = Z - np.max(Z, axis=-1, keepdims=True)
        exps = np.exp(shift_Z)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    def backward(self, dA):
        # the loss gradient goes through softmax without effect
        return dA

# loss calculator
class CrossEntropyLoss:
    def forward(self, Y_pred, Y_true):
        # calculate the error between the predictions and true targets
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-15)) / Y_pred.shape[0]
        return loss
    
    def backward(self):
        # calculate loss gradient
        return (self.Y_pred - self.Y_true) / self.Y_pred.shape[0]

# main neural network class
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        # forward propagation through all layers
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, loss_gradient):
        # backward propagation using chain rule
        for layer in self.layers[::-1]:
            loss_gradient = layer.backward(loss_gradient)

    def update_params(self, lr):
        # update weights and biases using learning rate
        for layer in self.layers:
            # check if layer is a hidden layer and not activation layer
            if hasattr(layer, 'W'):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

# accuracy calculator
def accuracy(Y_pred, Y_true):
    predicted_digits = np.argmax(Y_pred, axis=1)
    true_digits = np.argmax(Y_true, axis=1)
    return np.mean(predicted_digits == true_digits) * 100