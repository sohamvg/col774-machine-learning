import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import sys

solvers.options['show_progress'] = False

indicator = lambda exp: 1 if exp else 0

def gaussian_kernel(x, z, gamma):
    return np.exp(-gamma * np.linalg.norm(x-z)**2)

def extract_data(data):
    """
        Returns the feature data size m, feature vector x and label vector y.
    """
    m = data.shape[0]
    x = data[:, :-1]  # features
    x /= x.max()    # scale to 0-1
    y = data[:, -1].reshape((m, 1))  # labels

    return m, x, y

def get_class_data(data, class1, class2):
    """
        Returns data for class labels class1 and class2. Also changes the label class1 to -1 and class2 to 1 in the data.
    """
    data = data[(data[:,-1] == class1) | (data[:, -1] == class2)] # filter for class1 and class2
    data[:, -1] = np.where(data[:, -1] == class1, -1, 1) # change class labels to -1 and 1
    return data


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


class SVMBinaryClassifier:

    def __init__(self, train_data, C, gamma):
        self.m, self.x, self.y = extract_data(train_data)
        self.C = C
        self.gamma = gamma

    def train(self):
        m, x, y, C, gamma = self.m, self.x, self.y, self.C, self.gamma
        
        P = np.zeros((m, m))

        for i in range(m):
            for j in range(i, m):
                P[i, j] = y[i] * y[j] * gaussian_kernel(x[i], x[j], gamma)
                P[j, i] = P[i, j]

        q = -np.ones((m, 1))
        G = np.vstack((-np.identity(m), np.identity(m)))
        h = np.vstack((np.zeros((m, 1)), np.full((m, 1), C)))
        A = y.T
        b = np.zeros(1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        sol = solvers.qp(P,q,G,h,A,b)

        alpha = np.array(sol['x'])

        S = set() # support vectors

        # check for non-zero (>= epsilon) vectors
        epsilon = 1e-4
        while(len(S) == 0):
            for i in range(m):
                if alpha[i] >= epsilon:
                    S.add(i)
            epsilon = epsilon / 20

        b = sum(y[s] - sum(alpha[j] * y[j] * gaussian_kernel(x[j], x[s], gamma) for j in S) for s in S) / len(S) # take average over all support vectors

        self.alpha = alpha
        self.b = b

    def predict(self, x_predict):
        m, x, y, gamma, alpha, b = self.m, self.x, self.y, self.gamma, self.alpha, self.b
        return np.sign(sum(alpha[i] * y[i] * gaussian_kernel(x[i], x_predict, gamma) for i in range(m)) + b)


def run(train_data, test_data):
    train_data = np.genfromtxt(train_data, delimiter=',')
    test_data = np.genfromtxt(test_data, delimiter=',')

    total_classes = 10
    svm = [[None for j in range(total_classes)] for i in range(total_classes)]

    for i in range(total_classes):
        for j in range(i+1, total_classes):
            svm[i][j] = SVMBinaryClassifier(get_class_data(train_data, i, j), 1, 0.05)
            svm[i][j].train()

    m_test, x_test, y_test = extract_data(test_data)

    test_predictions = np.zeros(m_test, np.int)

    for t in range(m_test):
        votes = np.zeros(total_classes, np.int)
        for i in range(total_classes):
            for j in range(i+1, total_classes):
                p = svm[i][j].predict(x_test[t])
                if p < 0:
                    votes[i] += 1
                else:
                    votes[j] += 1
        test_predictions[t] = np.argmax(votes)

    return test_predictions

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()