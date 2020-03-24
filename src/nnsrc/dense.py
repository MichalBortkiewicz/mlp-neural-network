import numpy as np
import pandas as pd


class NeuralNetwork:
    """
    Whole network class
    """
    __slots__ = ['seed', 'n_layers', 'n_neurons_per_layer', 'act_funcs',
                 'bias', 'problem', 'error_function', 'error_function_deriv', 'layers', 'history']

    def __init__(self, n_layers, n_neurons_per_layer, act_funcs,
                 problem='classification', error_function=None, bias=True, seed=17):
        """Basic Neural Network configuration"""
        def problem_and_error_function():
            """Specify problem and error function"""
            if self.problem == 'classification_binary':
                if error_function is None or error_function == "binary_cross_entropy":
                    self.error_function = NeuralNetwork.binary_cross_entropy
                    self.error_function_deriv = NeuralNetwork.binary_crossentr_deriv
                elif error_function == "hinge":
                    self.error_function = NeuralNetwork.hinge
                    self.error_function_deriv = NeuralNetwork.hinge_deriv
                else:
                    raise Exception("Error function not know.")
            elif self.problem == 'regression':
                if error_function is None or error_function == "l2":
                    self.error_function = NeuralNetwork.l2_loss
                    self.error_function_deriv = NeuralNetwork.l2_loss_deriv
                elif error_function == "l1":
                    self.error_function = NeuralNetwork.l1_loss
                    self.error_function_deriv = NeuralNetwork.l1_loss_deriv
                else:
                    raise Exception("Error function not know.")
            elif self.problem == 'classification':
                if error_function is None or error_function == "cross_entropy":
                    self.error_function = NeuralNetwork.cross_entropy
                    self.error_function_deriv = NeuralNetwork.cross_entropy_deriv
                elif error_function == "sparse_cross_entropy":
                    pass
                else:
                    raise Exception("Error function not know.")
            else:
                raise Exception("Learning problem not known. Only classification and regression are valid options.")

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

        problem_and_error_function()

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
        __slots__ = ['input_dim', 'output_dim', 'act_func', 'name',
                     'weights', 'bias', 'activation', 'dactivation',
                     'use_bias', 'V_dW', 'V_db']

        def __init__(self, input_dim, output_dim, act_func, name, use_bias=True):

            # TODO: asserts
            self.name = name
            self.act_func = act_func
            self.output_dim = output_dim
            self.input_dim = input_dim
            self.use_bias = use_bias
            self.V_dW = 0
            self.V_db = 0

            if name == 'Dense_0':
                self.weights = np.eye(input_dim) # input and output dim should be the same in input layer
                self.bias = np.random.randn(self.output_dim, 1) * 0
            else:
                self.weights = np.random.rand(self.output_dim, self.input_dim) * 0.1
                self.bias = np.random.randn(self.output_dim, 1) * 0.1

            self.bias = self.bias.reshape((self.bias.shape[0], ))

            # region activation and deactivation
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
                self.dactivation = NeuralNetwork.stable_softmax_backward
            elif self.act_func == "tanh":
                self.activation = NeuralNetwork.tanh
                self.dactivation = NeuralNetwork.tanh_backward
            else:
                raise Exception('Non-supported activation function')
            # endregion

        def forward_propagation(self, A_prev):
            Z_curr = np.dot(self.weights, A_prev)
            if self.use_bias:
                Z_curr += self.bias[:, np.newaxis]
            return self.activation(Z_curr), Z_curr  # vectors

        def backward_propagation(self, dA_curr, Z_curr, A_prev):
            m = A_prev.shape[1]

            # all based on Andrew "Formulas for computing derivatives"

            # act func deriv
            dZ_curr = self.dactivation(dA_curr, Z_curr)
            # weights deriv
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            # bias deriv
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            # matrix A_prev deriv
            dA_prev = np.dot(self.weights.T, dZ_curr)

            return dA_prev, dW_curr, db_curr

        def update_weights(self, dW, db, alpha, beta):
            if 1 > beta > 0:
                self.V_dW = beta * self.V_dW + (1 - beta) * dW
                self.V_db = beta * self.V_db + (1 - beta) * db
                self.weights -= alpha * self.V_dW
                self.bias -= (alpha * self.V_db).reshape(self.bias.shape)

    def full_forward_propagation(self, X):
        cache = {}
        A_curr = X

        for i, layer in enumerate(self.layers):
            A_prev = A_curr

            A_curr, Z_curr = layer.forward_propagation(A_prev)

            cache["A" + str(i)] = A_curr
            cache["Z" + str(i)] = Z_curr

        return A_curr, cache

    def full_backward_propagation(self, Y_hat, Y, cache, n_classes=None):
        grads_values = {}

        # TODO: clean it up
        # loss function
        if self.problem == 'classification_binary':
            Y = Y.reshape(Y_hat.shape)
            dA_prev = self.error_function_deriv(Y_hat, Y)
        elif self.problem == 'regression':
            Y = Y.reshape(Y_hat.shape)
            dA_prev = self.error_function_deriv(Y_hat, Y)
        elif self.problem == 'classification':
            dA_prev = self.error_function_deriv(Y_hat, Y, n_classes)
        else:
            raise Exception("Learning problem not known. Only classification and regression are valid options.")

        for i, layer in reversed(list(enumerate(self.layers))):
            if i == 0:  # we don't want to update these weights
                break
            dA_curr = dA_prev
            A_prev = cache["A" + str(i-1)]
            Z_curr = cache["Z" + str(i)]

            dA_prev, dW_curr, db_curr = layer.backward_propagation(dA_curr, Z_curr, A_prev)
            grads_values["dA_curr" + str(i)] = dA_curr
            grads_values["dA_prev" + str(i)] = dA_prev
            grads_values["dW" + str(i)] = dW_curr
            grads_values["db" + str(i)] = db_curr

        return grads_values

    def predict(self, X):
        out, cache = self.full_forward_propagation(X)
        return out

    def train(self, X, Y, epochs, batch_size=None, alpha=0.01, beta=0.9, full_history=False, full_history_freq=1):
        n_classes = None

        # region asserts
        if self.problem == 'classification_binary':
            if self.error_function == 'binary_cross_entropy':
                assert NeuralNetwork.is_binary(Y), "Y values are not binary"
        elif self.problem == 'classification':
            assert np.min(Y) == 0, "There should be classes starting with 0 in multiclass classification problem"
            n_classes = np.max(Y) + 1
        elif self.problem == 'regression':
            pass
        # endregion

        self.history = {'cost': [], 'metrics': [], 'grads': [], 'caches': [], 'weights': [], 'biases': []}

        # for batch training
        X_true = np.copy(X)
        Y_true = np.copy(Y)
        if batch_size is not None:
            assert isinstance(batch_size, int), "batch_size is not an int"
            assert batch_size <= Y.shape[0], "batch_size larger than training dataset size"
            n_batches = int(Y.shape[0]/batch_size) + 1 if Y.shape[0] % batch_size != 0 \
                else int(Y.shape[0]/batch_size)
        else:
            n_batches = 1
            batch_size = Y.shape[0]

        for i in range(epochs):
            for j in range(n_batches):
                X = X_true[:, j*batch_size:(j+1)*batch_size] if j < n_batches-1 \
                    else X_true[:, j*batch_size:]  # well these indices should be the same
                Y = Y_true[j*batch_size:(j+1)*batch_size] if j < n_batches-1 \
                    else Y_true[j*batch_size:]     # but whatever...

                # forward
                Y_hat, cache = self.full_forward_propagation(X)

                # backward
                grads_values = self.full_backward_propagation(Y_hat, Y, cache, n_classes)

                # weight update
                self.update(grads_values, alpha, beta)

            # epoch history
            Y_hat, cache = self.full_forward_propagation(X_true)
            self.append_history_forward(Y_hat, Y_true, n_classes)
            grads_values = self.full_backward_propagation(Y_hat, Y_true, cache, n_classes)
            if full_history and i % full_history_freq == 0:
                self.append_history_backward(grads_values, cache)

        return self.history

    def update(self, grads_values, alpha, beta):
        # we are not updating input layer (W0) weights
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            layer.update_weights(grads_values["dW" + str(i)], grads_values["db" + str(i)], alpha, beta)

    def append_history_forward(self, Y_hat, Y, n_classes):
        # TODO: clean it up
        if self.problem == 'regression':
            self.history['cost'].append(self.error_function(Y_hat, Y))
            self.history['metrics'].append(self.r2_metric(Y_hat, Y))
        elif self.problem == 'classification_binary':
            self.history['cost'].append(self.error_function(Y_hat, Y))
            self.history['metrics'].append(NeuralNetwork.binary_accuracy(Y_hat, Y))
        elif self.problem == 'classification':
            self.history['cost'].append(self.error_function(Y_hat, Y, n_classes))
            self.history['metrics'].append(NeuralNetwork.multiclass_accuracy(Y_hat, Y, n_classes))

    def append_history_backward(self, grads_values, cache):
        self.history['grads'].append(grads_values)
        self.history['caches'].append(cache)
        self.history['weights'].append([l.weights.copy() for l in self.layers])
        self.history['biases'].append([l.bias.copy() for l in self.layers])

    # region activation functions
    @staticmethod
    def tanh(Z):
        return 2 / (1 + np.exp(-2*Z)) - 1

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
    # endregion

    # region activation_backward functions
    @staticmethod
    def stable_softmax_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        return dZ

    @staticmethod
    def tanh_backward(dA, Z):
        return dA * (1 - NeuralNetwork.tanh(Z)**2)

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
    # endregion

    @staticmethod
    def r2_metric(Y_hat, Y):
        pred = Y_hat.reshape(-1,1)
        true = Y.reshape(-1,1)
        return (1 - np.linalg.norm(pred - true) / (np.linalg.norm(true.mean() - true)))

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
        dl = np.sign(Y_hat - Y)
        return dl

    @staticmethod
    def hinge(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        return np.sum((np.maximum(0, 1 - Y_hat * Y)) / Y_hat.size)

    @staticmethod
    def hinge_deriv(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        return np.where(Y_hat * Y < 1, -Y / Y_hat.size, 0)

    @staticmethod
    def binary_crossentr_deriv(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        dl = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        return dl

    @staticmethod
    def cross_entropy_deriv(Y_hat, Y, n_classes):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        Y_one_hot = np.eye(n_classes)[Y]
        return Y_hat - Y_one_hot.T

    # TODO: modify a little bit this methods
    @staticmethod
    def binary_cross_entropy(Y_hat, Y):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    @staticmethod
    def cross_entropy(Y_hat, Y, n_classes):
        Y_hat = np.clip(Y_hat, 0.001, 0.999)
        m = Y_hat.shape[1]
        Y_one_hot = np.eye(n_classes)[Y]
        logprobs = np.dot(Y_one_hot, np.log(Y_hat))  # What is wrong with it???
        return -np.sum(logprobs) / m

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
    def binary_accuracy(Y_hat, Y):
        Y_hat_ = NeuralNetwork.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    @staticmethod
    def softmax_to_label(Y):
        Y_one_hot = NeuralNetwork.convert_softmax_into_class(Y)
        labels = [np.where(r == 1)[0][0] for r in Y_one_hot.T]
        return np.asanyarray(labels)

    @staticmethod
    def multiclass_accuracy(Y_hat, Y, n_classes=None):
        if n_classes is None:
            n_classes = np.max(Y) + 1
        Y_hat_one_hot = NeuralNetwork.convert_softmax_into_class(Y_hat)
        Y_one_hot = np.swapaxes(np.eye(n_classes)[Y], 0, 1)
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
# nn.train(X.T, y, 2000, 32, alpha=0.7)
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
# nn2.train(X.T, y, 2000, 32, 0.07)
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