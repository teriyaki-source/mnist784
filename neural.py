import numpy as np
import matplotlib.pyplot as plt

class ReLU_Classification_Neural_Network:
    """Usage:\n
    TO TRAIN the network:
    1) setup_training(...)
    2) gradient_descent(...)
    
    TO TEST the network:
    1) get_accuracy(make_predictions(x_test), y_test)
    
    TO INPUT own weights + biases:
    TODO"""
    def __init__(self):
        # TODO change this later to be variable based on user input
        # TODO make arrays of w, b values instead
        # Must be set using add_training_data in order to train the model, otherwise use TODO to set the weights
        self.weights = []
        self.biases = []
        self.Zs = []
        self.As = []
        self.delta_weights = []
        self.delta_biases = []
        self.delta_Zs = []
        self.num_layers = 0
        self.input_layer_size = 0
        self.output_layer_size = 0
        print("Created Neural Network")

    def setup_training(self, x, y, num_hidden_layers, hidden_layer_size, output_layer_size):     
        self.x_train = x
        self.y_train = y

        self.input_layer_size = x.shape[0]
        self.output_layer_size = output_layer_size
        self.num_layers = num_hidden_layers + 2

        self.weights = [[] for _ in range(num_hidden_layers + 1)]
        self.biases = [[] for _ in range(num_hidden_layers + 1)]
        self.Zs = [[] for _ in range(num_hidden_layers + 1)]
        self.As = [[] for _ in range(num_hidden_layers + 1)]

        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] = np.random.rand(hidden_layer_size, self.input_layer_size) - 0.5
                self.biases[i] = np.random.rand(hidden_layer_size, 1) - 0.5
                # self.Zs[i] = np.zeros(()) # don't know if this is needed?
            elif i == len(self.weights) - 1:
                self.weights[i] = np.random.rand(self.output_layer_size, hidden_layer_size) - 0.5
                self.biases[i] = np.random.rand(self.output_layer_size, 1) - 0.5
            else:
                self.weights[i] = np.random.rand(hidden_layer_size, hidden_layer_size) - 0.5
                self.biases[i] = np.random.rand(hidden_layer_size, 1) - 0.5

        self.delta_weights = [np.zeros_like(w) for w in self.weights]
        self.delta_biases = [np.zeros_like(b) for b in self.biases]
        self.delta_Zs = [np.zeros_like(z) if isinstance(z, np.ndarray) else None for z in self.Zs]

        print(f"Setting up network with: {num_hidden_layers} Hidden Layers of size {hidden_layer_size}.")


    def gradient_descent(self, iterations, alpha, save_values=False, reporting_frequency=100):
        print(f"Running {iterations} iterations, with learning rate {alpha}. Reporting every {reporting_frequency} iterations")
        self.alpha = alpha
        for i in range(iterations):
            self.forward_prop()
            self.backward_prop()
            self.update_params()
            if i % reporting_frequency == 0:
                print(f"Iterations: {i}")
                predictions = self.get_predictions(self.As[-1])
                print(f"{self.get_accuracy(predictions, self.y_train):.2f}% confidence on training data")

        if save_values:
            self.save_weights_biases()
        return self.weights, self.biases

    def forward_prop(self, x = None):
        if x is None:
            x = self.x_train
        for i in range(self.num_layers - 1):
            if i == 0:
                self.Zs[i] = self.weights[i].dot(x) + self.biases[i]
                self.As[i] = self.reLU(self.Zs[i])
            elif i == self.num_layers - 2:
                self.Zs[i] = self.weights[i].dot(self.As[i-1]) + self.biases[i]
                self.As[i] = self.softmax(self.Zs[i])
            else:
                self.Zs[i] = self.weights[i].dot(self.As[i-1]) + self.biases[i]
                self.As[i] = self.reLU(self.Zs[i])

    def backward_prop(self):
        # m is the number of training inputs
        m = self.x_train.shape[1]
        one_hot_y = self.one_hot(self.y_train)
        for i in range(self.num_layers -2, -1 , -1):
            if i == len(self.weights) - 1:
                self.delta_Zs[i] = self.As[i] - one_hot_y
                self.delta_weights[i] = 1 / m * self.delta_Zs[i].dot(self.As[i-1].T)
                self.delta_biases[i] = 1 / m * np.sum(self.delta_Zs[i], axis = 1, keepdims=True)
            elif i == 0:
                self.delta_Zs[i] = self.weights[i+1].T.dot(self.delta_Zs[i+1]) * self.reLU_deriv(self.Zs[i])
                self.delta_weights[i] = 1 / m * self.delta_Zs[i].dot(self.x_train.T)
                self.delta_biases[i] = 1 / m * np.sum(self.delta_Zs[i], axis = 1, keepdims=True)
            else:
                self.delta_Zs[i] = self.weights[i+1].T.dot(self.delta_Zs[i+1]) * self.reLU_deriv(self.Zs[i])
                self.delta_weights[i] = 1 / m * self.delta_Zs[i].dot(self.As[i-1].T)
                self.delta_biases[i] = 1 / m * np.sum(self.delta_Zs[i], axis = 1, keepdims=True)
    
    def reLU(self, z):
        return np.maximum(0, z)
    
    def reLU_deriv(self, z):
        return z > 0
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def softmax(self, z):
        z_shift = z - np.max(z, axis=0, keepdims=True)
        expZ = np.exp(z_shift)
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    def update_params(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * self.delta_weights[i]
            self.biases[i] -= self.alpha * self.delta_biases[i]
    
    def get_predictions(self, a):
        return np.argmax(a, 0)

    def get_accuracy(self, predictions, Y):
        # print(predictions, Y)
        return np.sum(predictions == Y) / Y.size * 100
    
    def make_predictions(self, x):
        self.forward_prop(x)
        predictions = self.get_predictions(self.As[-1])
        return predictions
    
    def test_prediction(self, x_input, y_input):        
        predictions = self.make_predictions(x_input)
        for i in range(x_input.shape[1]):  # loop over samples
            print(f"Prediction: {predictions[i]}")
            print(f"Label: {y_input[i]}")
            
            # Plot image
            current_image = x_input[:, i].reshape((28, 28)) * 255  # Reshape to 28x28 and scale to 255 for image
            plt.gray()
            plt.imshow(current_image, interpolation='nearest')
            plt.show()

# -----------------------------------------NOT FULLY IMPLEMENTED BELOW---------------------------------------------------------------------

    def save_weights_biases(self, weights_filename='weights', biases_filename='biases'):
        # Save each weight matrix to a text file
        for i, weight in enumerate(self.weights):
            np.savetxt(f"{weights_filename}_{i}.txt", weight)

        # Save each bias matrix to a text file
        for i, bias in enumerate(self.biases):
            np.savetxt(f"{biases_filename}_{i}.txt", bias)

    def load_weights_biases(self, weights_files, biases_files):
        """
        Load weights and biases from individual files and update the neural network.

        :param weights_files: List of filenames containing the weights for each layer.
        :param biases_files: List of filenames containing the biases for each layer.
        """
        # Ensure the number of files match for weights and biases
        assert len(weights_files) == len(biases_files), "Number of weight files and bias files must match"

        # Initialize the sizes of self.weights and self.biases
        self.weights = []
        self.biases = []

        # Load weights and biases from the files
        for i in range(len(weights_files)):
            # Load weight and bias from the respective files
            weight = np.loadtxt(weights_files[i])
            bias = np.loadtxt(biases_files[i])

            # Ensure the weight and bias arrays are correctly reshaped
            if i == 0:
                # First layer: weight shape should be (hidden_layer_size, input_layer_size)
                self.weights.append(weight.reshape(weight.shape[0], -1))  # Ensure the weight matrix is 2D
                self.biases.append(bias.reshape(-1, 1))  # Ensure bias is a column vector
            elif i == len(weights_files) - 1:
                # Last layer: weight shape should be (output_layer_size, hidden_layer_size)
                self.weights.append(weight.reshape(weight.shape[0], -1))
                self.biases.append(bias.reshape(-1, 1))
            else:
                # Hidden layers: weight shape should be (hidden_layer_size, hidden_layer_size)
                self.weights.append(weight.reshape(weight.shape[0], -1))
                self.biases.append(bias.reshape(-1, 1))
        
        # Optionally: Print the shapes of loaded weights and biases
        for i in range(len(self.weights)):
            print(f"Layer {i}: weight shape {self.weights[i].shape}, bias shape {self.biases[i].shape}")

        # Reinitialize the number of layers and input/output sizes
        self.num_layers = len(self.weights) + 1
        self.input_layer_size = self.weights[0].shape[1]  # First layer's input size
        self.output_layer_size = self.weights[-1].shape[0]  # Last layer's output size
