import numpy as np
import copy
import time


X_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_train.npy"
y_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_train.npy"
X_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_test.npy"
y_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_test.npy"

X_train, y_train = np.load(X_train_file), np.load(y_train_file)
X_test, y_test = np.load(X_test_file), np.load(y_test_file)


def one_hot_encode(y):
    ohe = np.zeros((y.size, y.max()+1))
    ohe[np.arange(y.size), y] = 1
    return ohe

m = X_train.shape[0]
n = 28 * 28
hidden_layers = [1]
r = 10
layers = hidden_layers.copy()
layers.append(r)

X_train = X_train.reshape((m, n)) # reshape
X_train = X_train / 255 # scale to 0-1
y_train = one_hot_encode(y_train)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def ReLU(z):
    return max(0.0, z)

def ReLU_derivative(z):
    if z > 0:
        return 1
    else:
        return 0

def netj(theta_j, x_j):
    return np.dot(theta_j.T, x_j)

def forward_propagation(x, theta, use_relu):
    """
        Arguments: 
            x: input example
            theta: parmeters
        Returns:
            o: outputs of each layer l and neuron j by forward propagation
    """
    o = [np.zeros(l) for l in layers]

    if use_relu:
        g = ReLU
    else:
        g = sigmoid

    for l in range(len(layers)):
        for j in range(layers[l]):
            if l == 0: # first layer: input is x from training data
                o[l][j] = g(netj(theta[l][j], x))
            elif l == len(layers) - 1: # output layer: use sigmoid always
                o[l][j] = sigmoid(netj(theta[l][j], o[l-1]))
            else: # hidden layers: input is output of prev layer
                o[l][j] = g(netj(theta[l][j], o[l-1])) # use all outputs of prev layer as network is fully connected
    
    return o


def back_propagation(y, o, theta, use_relu):
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
    for l in reversed(range(len(hidden_layers))):
        for j in range(hidden_layers[l]):
            if use_relu:
                derivative = ReLU_derivative(o[l][j])
            else:
                derivative = o[l][j] * (1 - o[l][j])
            deltas[l][j] = sum(deltas[l+1][dwn_nbr] * theta[l+1][dwn_nbr, j] * derivative for dwn_nbr in range(layers[l+1]))
        
    return deltas


def total_cost(theta, X, Y, use_relu):
    m = X.shape[0]
    error = 0
    for i in range(m):
        x, y = X[i], Y[i]
        o = forward_propagation(x, theta, use_relu)
        error += np.sum((y - o[-1]) ** 2)
    return error / (2 * m)


def init_theta(n, layers):
    # He initialization
    theta = [np.random.randn(layers[0], n) * np.sqrt(2/(n))] + [np.random.randn(layers[l], layers[l-1]) * np.sqrt(2/layers[l-1]) for l in range(1, len(layers))]
    return theta

def convergence_criteria(batch_size, m):
    """
        number of times to check for consecutive errors before converging based on batch_size and m (#training examples)
    """
    
    if 1 <= batch_size <= 100:
        k_repeats = 3
    elif batch_size == m:
        k_repeats = 1
    else:
        k_repeats = 2
    return k_repeats
    

def gradient_descent(X_train, y_train, M, learning_rate, epsilon, max_epochs, adaptive_learning, use_relu):
    """
        mini-batch SGD
        Arguments:
            M: batch size
            epsilon: tolerance for error
        Returns:
            optimal theta
    """
    epoch = 0
    k_repeats = 0
    k_repeats_limit = convergence_criteria(M, X_train.shape[0])
    theta = init_theta(n, layers)
    prev_cost = np.inf

    t0 = time.time()

    while True:
        epoch += 1
        if epoch > max_epochs:
            return theta

        if adaptive_learning:
            learning_rate = 0.5 / np.sqrt(epoch)

        print("epoch", epoch, total_cost(theta, X_train, y_train, use_relu), learning_rate, time.time() - t0)

        # shuffle at each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_train_e = X_train[indices]
        y_train_e = y_train[indices]

        for b in range(int(m/M)):
            sum_J_theta_derivatives = [np.zeros((layers[0], n))] + [np.zeros((layers[l], layers[l-1])) for l in range(1, len(layers))]

            for i in range(b * M, (b+1) * M):
                x, y = X_train_e[i], y_train_e[i]
                o = forward_propagation(x, theta, use_relu)
                deltas = back_propagation(y, o, theta, use_relu)

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
            cost = total_cost(theta, X_train_e[b * M: (b+1) * M], y_train_e[b * M: (b+1) * M], use_relu)
            if abs(prev_cost - cost) <= epsilon:
                k_repeats += 1
            else:
                k_repeats = 0

            if k_repeats >= k_repeats_limit:
                print("converged")
                return theta
            prev_cost = cost

            # update theta
            for l in range(len(layers)):
                    theta[l] = theta[l] - learning_rate * (sum_J_theta_derivatives[l] / M)

t0 = time.time()

theta_opt = gradient_descent(X_train, y_train, M=100, learning_rate=1, epsilon=5e-5, max_epochs=60, adaptive_learning=False, use_relu=False)

print("done", time.time() - t0)


# total_cost(theta_opt, X_train, y_train, use_relu=False)



################################3
# find accuracy

X_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_train.npy"
y_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_train.npy"
X_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_test.npy"
y_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_test.npy"

X_train, y_train = np.load(X_train_file), np.load(y_train_file)
X_test, y_test = np.load(X_test_file), np.load(y_test_file)

m = X_train.shape[0]
X_train = X_train.reshape((m, n)) # reshape
X_train = X_train / 255 # scale to 0-1

m_test = X_test.shape[0]
X_test = X_test.reshape((m_test, n)) # reshape
X_test = X_test / 255 # scale to 0-1

acc = 0
for i in range(m):
    o_pred = forward_propagation(X_train[i], theta_opt, use_relu=True)
    acc += int(np.argmax(o_pred[-1]) == y_train[i])

print("train", acc * 100/m)

acc = 0
for i in range(m_test):
    o_pred = forward_propagation(X_test[i], theta_opt, use_relu=True)
    acc += int(np.argmax(o_pred[-1]) == y_test[i])

print("test", acc * 100/m_test)

