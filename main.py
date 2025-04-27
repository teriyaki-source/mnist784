import numpy as np
import pandas as pd
import neural


# TODO make this more modular;
# [ ] store weights and biases if accuracy above certain %
# [ ] load weights and biases using a load_trained_data() function or similar
def main():
    data = pd.read_csv("mnist_test.csv")

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    # save a test set. each column is a number
    data_dev = data[:1000].T
    y_test = data_dev[0]
    x_test = data_dev[1:n] / 255

    # save training set. each column is a number
    data_train = data[1000:m].T
    y_train = data_train[0]
    x_train = data_train[1:n] / 255

    # create the neural network
    network = neural.ReLU_Classification_Neural_Network()

    # setup the training data. always run this first to setup the network
    network.setup_training(x_train, y_train, num_hidden_layers=1, hidden_layer_size=10, output_layer_size=10)

    # train the network using gradient descent
    network.gradient_descent(iterations=500, alpha=0.5, reporting_frequency=100)

    # test an input value
    network.test_prediction(x_test[:, 0:1], y_test[0:1])

    # TODO: implement a save_weights_biases() function to save the weights and biases to a file
    # weights = ['weights_0.txt', 'weights_1.txt']
    # biases = ['biases_0.txt', 'weights_1.txt']

    # network.load_weights_biases(weights, biases)

    # network.test_prediction(x_test[:, 0:1], y_test[0:1])

    # check the accuracy of the network on the test data
    print(f"{network.get_accuracy(network.make_predictions(x_test), y_test)}% accuracy on the test data.")

    return None

if __name__ ==  "__main__":
    main()