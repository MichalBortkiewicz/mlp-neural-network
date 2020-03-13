import numpy as np
import pandas as pd


class NeuralNetwork:
    """
    Whole network
    """
    __slots__ = ['n_in', 'n_out', 'seed', 'n_layers', 'n_neurons_per_layer', 'act_func',
                 'bias', 'n_batch', 'n_epochs', 'alpha', 'beta', 'problem', 'layers']

    def __init__(self, n_in, n_out, seed,  n_layers, n_neurons_per_layer, act_funcs, bias, n_batch,
                 n_epochs, alpha, beta, problem):
        """

        :param n_layers:
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

        self.n_in = n_in
        self.n_out = n_out
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
            input_dim = self.n_neurons_per_layer[i-1]
            output_dim = self.n_neurons_per_layer[i]
            act_func = self.act_func[i]
            name = "Dense_" + str(i)
            if i == 0:
                input_dim = self.n_in
            elif i == n_layers-1:
                output_dim = self.n_out
            self.layers.append(self.Dense(input_dim, output_dim, act_func, name))

    class Dense:
        """
        Dense (MLP) layer
        """
        __slots__ = ['input_dim', 'output_dim', 'act_func', 'weights', 'bias']

        def __init__(self, input_dim, output_dim, act_func, name):

            #TODO: asserts

            self.name = name
            self.act_func = act_func
            self.output_dim = output_dim
            self.input_dim = input_dim
            self.weights = np.random.rand(self.output_dim, self.input_dim) * 0.1
            self.bias = np.randoom.randn(self.output_dim, 1) * 0.1

    def forw_prop(self):
        pass

    def back_prop(self):
        pass

    def predict(self):
        pass

    def train(self):
       pass

