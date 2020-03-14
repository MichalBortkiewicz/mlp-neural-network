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

        #TODO: asserts


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
        __slots__ = ['input_dim', 'output_dim', 'act_func', 'name', 'weights', 'bias']

        def __init__(self, input_dim, output_dim, act_func, name):

            #TODO: asserts

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

    def forw_prop(self):
        pass

    def back_prop(self):
        pass

    def predict(self):
        pass

    def train(self):
       pass


#%%
nn = NeuralNetwork(seed=1, n_layers=3,
                   n_neurons_per_layer=[2, 4, 1], act_funcs='sigmoid', bias=True, n_batch=32,
                   n_epochs=10, alpha=0.007, beta=0.9, problem='classification')
