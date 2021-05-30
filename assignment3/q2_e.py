import numpy as np

X_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_train.npy"
y_train_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_train.npy"
X_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/X_test.npy"
y_test_file = "C:/IITD/sem5/col774-ml/datasets/kannada_digits/neural_network_kannada/y_test.npy"

X_train, y_train = np.load(X_train_file), np.load(y_train_file)
X_test, y_test = np.load(X_test_file), np.load(y_test_file)

m = X_train.shape[0]
n = 28 * 28

X_train = X_train.reshape((m, n)) # reshape
X_train = X_train / 255 # scale to 0-1

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,100), activation="relu", solver='sgd', learning_rate="adaptive", verbose=1)

m_test = X_test.shape[0]
X_test = X_test.reshape((m_test, n)) # reshape
X_test = X_test / 255 # scale to 0-1

print(clf.score(X_test, y_test))