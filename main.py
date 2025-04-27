import numpy as np
import pandas as pd

import neural

# followed from a youtube example
# it works!!!!
# TODO make this more modular;
# [ ] __init__ should no longer create any weight or bias attributes. create a new setup_trainig() function to input this
# [ ] variable input data size
# [ ] variable number of hidden layers
# [ ] variable hidden layer size
# [ ] variable output layer size
# [ ] store weights and biases if accuracy above certain %
# [ ] load weights and biases using a load_trained_data() function or similar
def main():
    data = pd.read_csv("mnist_test.csv")

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    #save a training set. each column is a number
    data_dev = data[:1000].T
    y_test = data_dev[0]
    x_test = data_dev[1:n] / 255

    data_train = data[1000:m].T
    y_train = data_train[0]
    x_train = data_train[1:n] / 255

    network = neural.ReLU_Classification_Neural_Network()

    network.setup_training(x_train, y_train, num_hidden_layers=1, hidden_layer_size=10, output_layer_size=10)

    network.gradient_descent(iterations=500, alpha=0.5, reporting_frequency=100)

    network.test_prediction(x_test[:, 0:1], y_test[0:1])

    # weights = ['weights_0.txt', 'weights_1.txt']
    # biases = ['biases_0.txt', 'weights_1.txt']

    # network.load_weights_biases(weights, biases)

    # network.test_prediction(x_test[:, 0:1], y_test[0:1])

    print(f"{network.get_accuracy(network.make_predictions(x_test), y_test)}% accuracy on the test data.")

    return None

if __name__ ==  "__main__":
    main()