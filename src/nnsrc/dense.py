import numpy as np
import pandas as pd


class NeuralNetwork:
    """
    Whole network
    """
    __slots__ = ['seed', 'n_layers', 'n_neurons_per_layer', 'act_funcs',
                 'bias', 'n_batch', 'n_epochs', 'alpha', 'beta', 'problem', 'layers']

    def __init__(self, seed,  n_layers, n_neurons_per_layer,
                 act_funcs, bias, n_batch,
                 n_epochs, alpha, beta, problem):
        """

        :param n_layers: number of layers (in + hidden*x + out)
        :param n_neurons_per_layer:
        :param act_func:
        :param bias:
        :param n_batch:
        :param n_epochs:
        :param alpha:
        :param beta:
        :param problem:
        """

        # TODO: asserts

        self.seed = seed
        self.problem = problem
        self.beta = beta
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.bias = bias
        self.act_funcs = act_funcs
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.layers = []

        np.random.seed(seed)

        for i in range(n_layers):
            output_dim = self.n_neurons_per_layer[i]
            act_func = self.act_funcs[i]
            name = "Dense_" + str(i)
            if i == 0:
                input_dim = self.n_neurons_per_layer[i]
            else:
                input_dim = self.n_neurons_per_layer[i - 1]

            self.layers.append(self.Dense(input_dim, output_dim, act_func, name))

    class Dense:
        """
        Dense (MLP) layer
        """
        __slots__ = ['input_dim', 'output_dim', 'act_func', 'name', 'weights', 'bias', 'activation', 'dactivation']

        def __init__(self, input_dim, output_dim, act_func, name):

            # TODO: asserts
            self.name = name
            self.act_func = act_func
            self.output_dim = output_dim
            self.input_dim = input_dim

            if name == 'Dense_0':
                self.weights = np.eye(input_dim) # input and output dim should be the same in input layer
                self.bias = np.random.randn(self.output_dim, 1) * 0
            else:
                self.weights = np.random.rand(self.output_dim, self.input_dim) * 0.1
                self.bias = np.random.randn(self.output_dim, 1) * 0.1

            self.bias = self.bias.reshape((self.bias.shape[0], ))

            self.activation = NeuralNetwork.relu
            self.dactivation = NeuralNetwork.relu_backward
            if self.act_func is "relu":
                self.activation = NeuralNetwork.relu
                self.dactivation = NeuralNetwork.relu_backward
            elif self.act_func is "sigmoid":
                self.activation = NeuralNetwork.sigmoid
                self.dactivation = NeuralNetwork.sigmoid_backward
            else:
                raise Exception('Non-supported activation function')



        def forward_propagation(self, A_prev):
            Z_curr = np.dot(self.weights, A_prev) + self.bias
            return self.activation(Z_curr), Z_curr  # vectors

        def backward_propagation(self, dA_curr, Z_curr, A_prev):
            m = A_prev.shape[1]

            # all based on Andrew "Formulas for computing derivatives"
            # act func deriv
            dZ_curr = self.dactivation(dA_curr, Z_curr)  # TODO: maybe do it more elegant?
            # weights deriv
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            # bias deriv
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            # matrix A_prev deriv
            dA_prev = np.dot(self.weights.T, dZ_curr)

            return dA_prev, dW_curr, db_curr

    def full_forward_propagation(self, X):
        cache = {}
        A_curr = X

        for i, layer in enumerate(self.layers):
            A_prev = A_curr

            A_curr, Z_curr = layer.forward_propagation(A_prev)

            cache["A" + str(i)] = A_curr
            cache["Z" + str(i)] = Z_curr

        return A_curr, cache


    def full_backward_propagation(self, Y_hat, Y, cache):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        # for binary crossentropy (classification) - TODO: modify this step
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for i, layer in reversed(list(enumerate(self.layers))):
            dA_curr = dA_prev

            A_prev = cache["A" + str(i-1)]
            Z_curr = cache["Z" + str(i)]

            dA_prev, dW_curr, db_curr = layer.backward_propagation(dA_curr, Z_curr, A_prev)
            grads_values["dW" + str(i)] = dW_curr
            grads_values["db" + str(i)] = db_curr

        return grads_values

    def predict(self):
        pass

    def train(self):
        pass


    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid_backward(dA, Z):
        sig = NeuralNetwork.sigmoid(Z)
        return dA * sig * (1 - sig)

    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

#%%
nn = NeuralNetwork(seed=1, n_layers=3,
                   n_neurons_per_layer=[2, 4, 1], act_funcs=['sigmoid', 'sigmoid', 'sigmoid'], bias=True, n_batch=32,
                   n_epochs=10, alpha=0.007, beta=0.9, problem='classification')


