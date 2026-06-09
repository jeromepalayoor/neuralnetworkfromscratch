import numpy as np

beta = 0.9

# hidden layer
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # initialise weights and biases
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)

    def forward(self, X):
        # forward propagation
        Z = np.dot(X, self.W) + self.b
        self.X = X
        return Z
    
    def backward(self, dZ):
        # backward propagation
        # get the loss gradient and use chain rule to find the parameter gradients
        n = self.X.shape[0]
        dW = np.dot(self.X.T, dZ) / n
        db = np.sum(dZ, axis=0, keepdims=True) / n
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

# sigmoid activation layer
class SigmoidLayer:
    # basic sigmoid function output is between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        return self.sigmoid(x)
    
    def backward(self, loss_gradient):
        # calculate derivative
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x)) * loss_gradient

# cross entropy loss calculator
class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        # calculate the error between the predictions and true targets
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / y_pred.shape[0]
        return loss
    
    def backward(self):
        # calculate loss gradient
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]

# binary cross entropy loss calculator
class BCELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        return -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))

    def backward(self):
        return (1 / self.y_pred.shape[0]) * (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) + 1e-15)

# mean square error loss calculator
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
        
    def backward(self):
        N = self.y_pred.shape[0]
        return (2 / N) * (self.y_pred - self.y_true)

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
                # momentum optization
                layer.v_W = beta * layer.v_W + (1 - beta) * layer.dW
                layer.W -= lr * layer.v_W
                layer.v_b = beta * layer.v_b + (1 - beta) * layer.db
                layer.b -= lr * layer.v_b

# accuracy calculator
def accuracy(Y_pred, Y_true):
    predicted_digits = np.argmax(Y_pred, axis=1)
    true_digits = np.argmax(Y_true, axis=1)
    return np.mean(predicted_digits == true_digits) * 100
