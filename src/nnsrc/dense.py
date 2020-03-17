import numpy as np
import pandas as pd


class NeuralNetwork:
    """
    Whole network
    """
    __slots__ = ['seed', 'n_layers', 'n_neurons_per_layer',
                 'act_funcs', 'bias', 'problem', 'layers', 'history']

    def __init__(self, n_layers, n_neurons_per_layer, act_funcs,
                 bias=True, problem='classification', seed=17):
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

        self.history = None
        self.seed = seed
        self.problem = problem
        self.bias = bias
        self.act_funcs = act_funcs
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.layers = []

        np.random.seed(seed)

        for i in range(n_layers):
            output_dim = self.n_neurons_per_layer[i]
            act_func = self.act_funcs[i] if type(self.act_funcs) is list else self.act_funcs
            name = "Dense_" + str(i)
            if i == 0:
                input_dim = self.n_neurons_per_layer[i]
            else:
                input_dim = self.n_neurons_per_layer[i - 1]

            self.layers.append(self.Dense(input_dim, output_dim, act_func, name, bias))

    class Dense:
        """
        Dense (MLP) layer
        """
        __slots__ = ['input_dim', 'output_dim', 'act_func', 'name', 'weights', 'bias', 'activation', 'dactivation', 'use_bias']

        def __init__(self, input_dim, output_dim, act_func, name, use_bias=True):

            # TODO: asserts
            self.name = name
            self.act_func = act_func
            self.output_dim = output_dim
            self.input_dim = input_dim
            self.use_bias = use_bias

            if name == 'Dense_0':
                self.weights = np.eye(input_dim) # input and output dim should be the same in input layer
                self.bias = np.random.randn(self.output_dim, 1) * 0
            else:
                self.weights = np.random.rand(self.output_dim, self.input_dim) * 0.1
                self.bias = np.random.randn(self.output_dim, 1) * 0.1

            self.bias = self.bias.reshape((self.bias.shape[0], ))

            self.activation = NeuralNetwork.relu
            self.dactivation = NeuralNetwork.relu_backward
            if self.act_func == "relu":
                self.activation = NeuralNetwork.relu
                self.dactivation = NeuralNetwork.relu_backward
            elif self.act_func == "sigmoid":
                self.activation = NeuralNetwork.sigmoid
                self.dactivation = NeuralNetwork.sigmoid_backward
            elif self.act_func == "linear":
                self.activation = NeuralNetwork.linear
                self.dactivation = NeuralNetwork.linear_backward
            elif self.act_func == "softmax":
                self.activation = NeuralNetwork.stable_softmax

            else:
                raise Exception('Non-supported activation function')

        def forward_propagation(self, A_prev):
            Z_curr = np.dot(self.weights, A_prev)
            if self.use_bias:
                Z_curr += self.bias[:, np.newaxis]
            return self.activation(Z_curr), Z_curr  # vectors

        def backward_propagation(self, dA_curr, Z_curr, A_prev):
            m = A_prev.shape[1]

            # all based on Andrew "Formulas for computing derivatives"
            if self.act_func == "softmax":  # TODO: is it all right?
                dZ_curr = dA_curr
            else:
                # act func deriv
                dZ_curr = self.dactivation(dA_curr, Z_curr)  # TODO: maybe do it more elegant?
                # weights deriv
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            # bias deriv
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            # matrix A_prev deriv
            dA_prev = np.dot(self.weights.T, dZ_curr)

            return dA_prev, dW_curr, db_curr

        def update_weights(self, dW, db, alpha):
            self.weights -= alpha * dW
            self.bias -= (alpha * db).reshape(self.bias.shape)

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

        # loss function
        if self.problem == 'classification_binary':
            Y = Y.reshape(Y_hat.shape)
            dA_prev = NeuralNetwork.bin_crossentr_deriv(Y_hat, Y)
        elif self.problem == 'regression':
            Y = Y.reshape(Y_hat.shape)
            dA_prev = NeuralNetwork.l2_loss_deriv(Y_hat, Y)
        elif self.problem == 'classification':
            dA_prev = NeuralNetwork.crossentr_deriv(Y_hat, Y)
        else:
            raise Exception("Learning problem not known. Only classification and regression are valid options.")

        for i, layer in reversed(list(enumerate(self.layers))):
            if i == 0:  # we don't want to update these weights
                break
            dA_curr = dA_prev
            A_prev = cache["A" + str(i-1)]
            Z_curr = cache["Z" + str(i)]

            dA_prev, dW_curr, db_curr = layer.backward_propagation(dA_curr, Z_curr, A_prev)
            grads_values["dA_curr" + str(i)] = dA_curr-
            grads_values["dA_prev" + str(i)] = dA_prev
            grads_values["dW" + str(i)] = dW_curr
            grads_values["db" + str(i)] = db_curr

        return grads_values

    def predict(self, X):
        out, cache = self.full_forward_propagation(X)
        # if self.problem == 'classification':
        #     out = NeuralNetwork.one_hot_to_label(out)
        return out

    def train(self, X, Y, epochs, alpha=0.01, beta=None, full_history=False):
        # asserts
        if self.problem == 'classification_binary':
            assert NeuralNetwork.is_binary(Y), "Y values are not binary"
        elif self.problem == 'classification':
            pass
        elif self.problem == 'regression':
            pass

        self.history = {'cost': [], 'metrics': [],
                        'grads': [],
                        'caches': [],
                        'weights': [],
                        'biases': []}


        for i in range(epochs):
            # forward
            Y_hat, cache = self.full_forward_propagation(X)

            if self.problem == 'regression':
                self.history['cost'].append(NeuralNetwork.l2_loss_deriv(Y_hat, Y))
                self.history['metrics'].append(NeuralNetwork.l2_loss(Y_hat, Y))
            elif self.problem == 'classification_binary':
                self.history['cost'].append(NeuralNetwork.binary_cross_entropy_cost(Y_hat, Y))
                self.history['metrics'].append(NeuralNetwork.get_binary_accuracy_value(Y_hat, Y))
            elif self.problem == 'classification':
                self.history['cost'].append(NeuralNetwork.cross_entropy_cost(Y_hat, Y))
                self.history['metrics'].append(NeuralNetwork.get_multiclass_accuracy(Y_hat, Y))

            # backward
            grads_values = self.full_backward_propagation(Y_hat, Y, cache)
            if full_history:
                self.history['grads'].append(grads_values)
                self.history['caches'].append(cache)
                self.history['weights'].append([l.weights.copy() for l in self.layers])
                self.history['biases'].append([l.bias.copy() for l in self.layers])
            self.update(grads_values, alpha)
        return self.history

    def update(self, grads_values, alpha):
        # we are not updating input layer (W0) weights
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            layer.update_weights(grads_values["dW" + str(i)], grads_values["db" + str(i)], alpha)


    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def linear(Z):
        return Z.copy()

    @staticmethod
    def stable_softmax(Z):
        exps = np.exp(Z - np.max(Z))
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def cross_entropy_cost(X, y):
        Y = y.reshape((1, y.shape[0]))
        m = Y.shape[1]
        logprobs = np.multiply(np.log(X), Y) + np.multiply((1 - Y), np.log(1 - X))
        return - np.sum(logprobs) / m

    @staticmethod
    def sigmoid_backward(dA, Z):
        sig = NeuralNetwork.sigmoid(Z)
        return dA * sig * (1 - sig)

    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def linear_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        return dZ

    @staticmethod
    def l2_loss(Y_hat, Y):
        return np.linalg.norm(Y_hat-Y)

    @staticmethod
    def l2_loss_deriv(Y_hat, Y):
        dl = (Y_hat-Y)
        return dl

    @staticmethod
    def l1_loss(Y_hat, Y):
        return np.linalg.norm(Y_hat - Y, 1)

    @staticmethod
    def l1_loss_deriv(Y_hat, Y):
        dl = (Y_hat - Y)
        return dl

    @staticmethod
    def bin_crossentr_deriv(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        dl = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        return dl

    @staticmethod
    def crossentr_deriv(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        Y_one_hot = np.eye(np.max(Y) + 1)[Y]
        return Y_hat - Y_one_hot.T

    # TODO: modify a little bit this methods
    @staticmethod
    def binary_cross_entropy_cost(Y_hat, Y):
        # number of examples
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    @staticmethod
    def convert_prob_into_class(probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    @staticmethod
    def convert_softmax_into_class(probs):
        one_hot = np.copy(probs)
        one_hot[one_hot == np.max(one_hot, axis=0)] = 1
        one_hot[one_hot < 1] = 0
        return one_hot

    @staticmethod
    def get_binary_accuracy_value(Y_hat, Y):
        Y_hat_ = NeuralNetwork.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    @staticmethod
    def one_hot_to_label(Y):
        Y_one_hot = NeuralNetwork.convert_softmax_into_class(Y)
        labels = [np.where(r==1)[0][0] for r in Y_one_hot.T]
        return np.asanyarray(labels)

    @staticmethod
    def get_multiclass_accuracy(Y_hat, Y):
        Y_hat_one_hot = NeuralNetwork.convert_softmax_into_class(Y_hat)
        Y_one_hot = np.swapaxes(np.eye(np.max(Y) + 1)[Y], 0, 1)
        return np.count_nonzero(np.sum(abs(Y_hat_one_hot-Y_one_hot), axis=0) == 0) / Y_hat_one_hot.shape[1]

    @staticmethod
    def is_binary(Y):
        uniques = np.unique(Y)
        if len(uniques) == 2:
            if set(uniques).issubset([0, 1]):
                return True
        return False



## classification_binary
# data = pd.read_csv('../data/classification/data.simple.test.100.csv')
#
# X = data[["x", "y"]].values
# y = data["cls"].values
#
# nn = NeuralNetwork(seed=1, n_layers=4,
#                     n_neurons_per_layer=[2, 10,  100, 1], act_funcs=['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
#                     bias=True, problem='classification_binary')
#
# for layer in nn.layers:
#     print(layer.name, layer.input_dim, layer.output_dim)
#
# y = y-1  # for binary crossentropy
# nn.train(X.T, y, 20000, 0.7)
#
# print("CLASSIFICATION DONE")
# y_hat = nn.predict(X.T)
#
#
# ## classification
# data = pd.read_csv('../data/classification/data.three_gauss.train.500.csv')
#
# X = data[["x", "y"]].values
# y = data["cls"].values
#
# nn2 = NeuralNetwork(seed=1, n_layers=4,
#                     n_neurons_per_layer=[2, 10,  100, 3], act_funcs=['sigmoid', 'sigmoid', 'sigmoid', 'softmax'],
#                     bias=True, problem='classification')
#
#
# for layer in nn2.layers:
#     print(layer.name, layer.input_dim, layer.output_dim)
#
# y = y-1
# nn2.train(X.T, y, 2000, 0.7)
#
# print("CLASSIFICATION DONE")
# y_hat = nn2.predict(X.T)


## regression
# data = pd.read_csv('../data/regression/data.activation.train.100.csv')
# X = data[["x"]].values
# y = data["y"].values
#
# nn2 = NeuralNetwork(seed=1, n_layers=4,
#                     n_neurons_per_layer=[1, 100,  100, 1], act_funcs=['relu', 'relu', 'relu', 'linear'],
#                     bias=True, problem='regression')
#
# for layer in nn2.layers:
#     print(layer.name, layer.input_dim, layer.output_dim)
#
# nn2.train(X.T, y, 20000, 0.07)
#
# data = pd.read_csv('../data/regression/data.activation.test.100.csv')
# X = data[["x"]].values
# y = data['y'].values
#
# y_hat = nn2.predict(X.T)



## debugging examples
"""
nn = NeuralNetwork(seed=1, n_layers=3,
                   n_neurons_per_layer=[2, 4, 1], act_funcs=['sigmoid', 'sigmoid', 'sigmoid'], bias=True,
                   # n_neurons_per_layer=[2, 4, 1], act_funcs=['relu', 'relu', 'relu'], bias=True,
                   problem='classification')
"""

# nn.train(np.asanyarray([[0, 1], [0, 2], [1, 0]]).T, np.asanyarray([1, 1, 0]).T.reshape((3, )), epochs=1000, alpha=0.1)
# nn.train(np.asanyarray([[0, 1], [0, 2], [1, 0], [-0.2, 1.5]]).T, np.asanyarray([1, 1, 0, 1]).T.reshape((4, )), epochs=100000, alpha=0.01)
# nn.train(np.asanyarray([[0.2, 0.1], [0.1, 0.7]]).T, np.asanyarray([1, 0]).T.reshape((2, )), epochs=1000, alpha=0.1)
# nn.train(np.asanyarray([[0.2, 0.1]]).T, np.asanyarray([1]).T.reshape((1, )), epochs=1000, alpha=0.1)