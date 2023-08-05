import numpy as np


class NN:
    def __init__(self):
        # create and randomly initialize weights:
        self.weights_0 = 2 * np.random.random((3, 4)) - 1
        self.weights_1 = 2 * np.random.random((4, 8)) - 1
        self.weights_2 = 2 * np.random.random((8, 1)) - 1

    def sigmoid_function(self, x, derivative=False):

        if derivative:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def train(self, X_train, y_train, num_iterations):

        print(f'Training Neural Network with {num_iterations} iterations.')

        for it in range(0, num_iterations):

            input_layer = X_train
            layer_1 = self.sigmoid_function(np.dot(input_layer, self.weights_0))
            layer_2 = self.sigmoid_function(np.dot(layer_1, self.weights_1))
            output_layer = self.sigmoid_function(np.dot(layer_2, self.weights_2))

            # calculate error
            output_error = y_train - output_layer
            output_delta = output_error * self.sigmoid_function(output_layer, derivative=True)

            if (it % 2000 == 0):
                print(f'Error rate: {round(np.mean(np.abs(output_error)), 4)}')

            # backpropagation for layer_2 and layer_1
            layer_2_error = np.dot(output_delta, self.weights_2.T)
            layer_2_delta = layer_2_error * self.sigmoid_function(layer_2, derivative=True)

            layer_1_error = np.dot(layer_2_delta, self.weights_1.T)
            layer_1_delta = layer_1_error * self.sigmoid_function(layer_1, derivative=True)

            # update weights
            self.weights_0 += np.dot(input_layer.T, layer_1_delta)
            self.weights_1 += np.dot(layer_1.T, layer_2_delta)
            self.weights_2 += np.dot(layer_2.T, output_delta)

        return output_layer

    def predict(self, inp):

        l1 = self.sigmoid_function(np.dot(inp, self.weights_0))
        l2 = self.sigmoid_function(np.dot(l1, self.weights_1))

        return self.sigmoid_function(np.dot(l2, self.weights_2))


if __name__ == '__main__':
    # seed for reproducibility
    np.random.seed(8)

    # create neural network instance
    neural_net = NN()

    # create training data
    input_X_train = np.random.randint(0, 2, size=(10, 3))
    output_y_train = np.random.randint(0, 2, size=(10, 1))

    training_output = neural_net.train(input_X_train, output_y_train, 20000)
    # print(training_output)

    # Predicting for example
    example = [1, 0, 1]
    prediction = neural_net.predict(example)
    print(f'Input: {example} Prediction: {prediction}')
