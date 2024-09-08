import numpy as np
import mnist_loader as ml


class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # skip first layer, as it has no weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # assume that all neurons from layer before, are connected to all neurons in the next layer
        # skip last layer, because it doesn't have any output connections
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def train(
        self,
        training_data,
        test_data,
        num_epochs,
        mini_batch_size=10,
        learning_rate=3.0,
    ):
        for epoch in range(num_epochs):
            # select mini-batch
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            print(f"Epoch {epoch} complete")
            result = round(self.test(test_data) / len(test_data) * 100, 2)
            print(f"Accuracy: {result}%")

    def test(self, test_data):
        return self.evaluate(test_data)

    def update_mini_batch(self, mini_batch, learning_rate):
        output_weights = [np.zeros(w.shape) for w in self.weights]
        output_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            cost_function_gradient_w, cost_function_gradient_b = self.backpropagation(
                x, y
            )

            output_weights = [
                ow + cw for ow, cw in zip(output_weights, cost_function_gradient_w)
            ]
            output_biases = [
                ob + cb for ob, cb in zip(output_biases, cost_function_gradient_b)
            ]

        self.weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, output_weights)
        ]

        self.biases = [
            b - (learning_rate / len(mini_batch)) * nb
            for b, nb in zip(self.biases, output_biases)
        ]

    def backpropagation(self, x, y):
        # initialize the gradients of the biases and weights to zero
        cost_function_gradient_w = [np.zeros(w.shape) for w in self.weights]
        cost_function_gradient_b = [np.zeros(b.shape) for b in self.biases]

        # feed forward
        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)

        # backward pass
        delta = (activations[-1] - y * 2) * derivative_sigmoid(zs[-1])
        cost_function_gradient_b[-1] = delta
        cost_function_gradient_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = derivative_sigmoid(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            cost_function_gradient_b[-layer] = delta
            cost_function_gradient_w[-layer] = np.dot(
                delta, activations[-layer - 1].transpose()
            )
        return cost_function_gradient_w, cost_function_gradient_b

    def feed_forward(self, x):
        # calculate the output of the network
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
