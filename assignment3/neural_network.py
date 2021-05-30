import sys
import os
import numpy as np


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def one_hot_encode(y):
    ohe = np.zeros((y.size, y.max()+1))
    ohe[np.arange(y.size), y] = 1
    return ohe

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return 1 * (z > 0) # 1 for z[i] > 0 else 0 for each i

def net(theta, x):
    """
        Returns net_j = sum(theta_j . x_j)
    """
    return theta.dot(x)

def forward_propagation(x, theta, only_output_layer, use_relu, layers):
    """
        Arguments: 
            x: input example
            theta: parmeters
            only_output_layer: if True, return only output of last layer else return output for all layers
        Returns:
            o: outputs of each layer l and neuron j by forward propagation
    """
    
    o = [np.zeros(l) for l in layers]

    if use_relu:
        g = ReLU
    else:
        g = sigmoid

    for l in range(len(layers)):
        if l == 0: # first layer: input is x from training data
            o[l] = g(net(theta[l], x))
        elif l == len(layers) - 1: # output layer: use sigmoid always
            o[l] = sigmoid(net(theta[l], o[l-1]))
        else: # hidden layers: input is output of prev layer
            o[l] = g(net(theta[l], o[l-1]))

    if only_output_layer:
        return o[-1]
    else:
        return o

def back_propagation(y, o, theta, use_relu, layers):
    """
        Arguments:
            y: class labels
            o: outputs of each layer
            theta: parameters
        Returns:
            deltas: deltas[l][j] for each layer l and neuron j by backpropagation
    """
    deltas = [np.zeros(l) for l in layers]

    # output layer
    output_layer = -1
    delta = (y - o[output_layer]) * o[output_layer] * (1 - o[output_layer])
    deltas[output_layer] = delta

    # hidden layers
    for l in reversed(range(len(layers) - 1)):
        if use_relu:
            derivative = ReLU_derivative(o[l])
        else:
            derivative = o[l] * (1 - o[l]) # equivalent to derivative of sigmoid(netj)

        deltas[l] = (theta[l+1].T @ deltas[l+1]) * derivative # = sum(deltas[l+1][dwn_nbr] * theta[l+1][dwn_nbr, j] * derivative for dwn_nbr in range(layers[l+1]))
        
    return deltas


def get_cost(theta, X, Y, use_relu, layers):
    m = X.shape[0]
    outputs = np.apply_along_axis(forward_propagation, 1, X, theta, True, use_relu, layers)
    return np.sum((outputs - Y) ** 2) / (2 * m)


def init_theta(n, layers):
    # He initialization
    theta = [np.random.randn(layers[0], n) * np.sqrt(2/(n))] + [np.random.randn(layers[l], layers[l-1]) * np.sqrt(2/layers[l-1]) for l in range(1, len(layers))]
    return theta

def convergence_criteria(batch_size, m):
    """
        number of times to check for consecutive errors before converging based on batch_size and m (#training examples)
    """
    
    if 1 <= batch_size <= 50:
        k_repeats = 3
    elif batch_size == m:
        k_repeats = 1
    else:
        k_repeats = 2
    return k_repeats
    
def gradient_descent(X_train, y_train, M, learning_rate, epsilon, max_epochs, adaptive_learning, use_relu, layers):
    """
        mini-batch SGD
    """

    n = 28 * 28
    m = X_train.shape[0]

    epoch = 0
    k_repeats = 0
    k_repeats_limit = convergence_criteria(M, m)

    theta = init_theta(n, layers)
    prev_cost = np.inf

    while True:
        epoch += 1
        if epoch > max_epochs:
            return theta

        if adaptive_learning:
            learning_rate = 0.5 / np.sqrt(epoch)

        # print("e", epoch, get_cost(theta, X_train, y_train, use_relu, layers))

        # shuffle at each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_train_e = X_train[indices]
        y_train_e = y_train[indices]

        for b in range(int(m/M)):
            sum_J_theta_derivatives = [np.zeros((layers[0], n))] + [np.zeros((layers[l], layers[l-1])) for l in range(1, len(layers))]

            for i in range(b * M, (b+1) * M):
                x, y = X_train_e[i], y_train_e[i]
                o = forward_propagation(x, theta, only_output_layer=False, use_relu=use_relu, layers=layers)
                deltas = back_propagation(y, o, theta, use_relu, layers=layers)

                # calculate J(theta) derivatives
                for l in range(len(layers)):
                    if l == 0:
                        x_j = x
                    else:
                        x_j = o[l-1]
                    for j in range(layers[l]):
                        J_theta_derivative = - deltas[l][j] * x_j
                        sum_J_theta_derivatives[l][j] += J_theta_derivative # sum over J(theta) derivatives over the batch

            # calculating cost over the examples seen in the lastest batch before updating theta
            cost = get_cost(theta, X_train_e[b * M: (b+1) * M], y_train_e[b * M: (b+1) * M], use_relu, layers)
            if abs(prev_cost - cost) <= epsilon:
                k_repeats += 1
            else:
                k_repeats = 0

            if k_repeats >= k_repeats_limit:
                # print("converged")
                return theta
            prev_cost = cost

            # update theta
            for l in range(len(layers)):
                    theta[l] = theta[l] - learning_rate * (sum_J_theta_derivatives[l] / M)



def predict(theta, X, use_relu, layers):
    outputs = np.apply_along_axis(forward_propagation, 1, X, theta, True, use_relu, layers)
    predictions = np.argmax(outputs, axis=1)
    return predictions


def run(X_train_file, y_train_file, X_test_file, batch_size, hidden_layer_list, activation):

    X_train, y_train = np.load(X_train_file), np.load(y_train_file)
    X_test = np.load(X_test_file)

    m = X_train.shape[0]
    n = 28 * 28
    hidden_layers = hidden_layer_list
    r = 10
    layers = hidden_layers.copy()
    layers.append(r)

    X_train = X_train.reshape((m, n)) # reshape
    X_train = X_train / 255 # scale to 0-1
    y_train = one_hot_encode(y_train)

    if activation == "relu":
        use_relu = True
    else:
        use_relu = False

    # print(m, n, hidden_layers, batch_size, use_relu)

    theta_opt = gradient_descent(X_train, y_train, M=batch_size, learning_rate=0.4, epsilon=2e-4, max_epochs=80, adaptive_learning=False, use_relu=use_relu, layers=layers)

    m_test = X_test.shape[0]
    X_test = X_test.reshape((m_test, n)) # reshape
    X_test = X_test / 255 # scale to 0-1

    predictions = predict(theta_opt, X_test, use_relu=use_relu, layers=layers)
    
    return predictions


def main():
    x_train = sys.argv[1]
    y_train = sys.argv[2]
    x_test = sys.argv[3]
    output_file = sys.argv[4]
    batch_size = int(sys.argv[5])
    hidden_layer_list = [int(i) for i in sys.argv[6].split()]
    activation = sys.argv[7]

    output = run(x_train, y_train, x_test, batch_size, hidden_layer_list, activation)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
