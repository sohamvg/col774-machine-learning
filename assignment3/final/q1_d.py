import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# load data
train_file = "C:/IITD/sem5/col774-ml/datasets/decision_tree/decision_tree/train.csv"
test_file = "C:/IITD/sem5/col774-ml/datasets/decision_tree/decision_tree/test.csv"
val_file = "C:/IITD/sem5/col774-ml/datasets/decision_tree/decision_tree/val.csv"
train_data = np.genfromtxt(train_file, delimiter=',')
test_data = np.genfromtxt(test_file, delimiter=',')
val_data = np.genfromtxt(val_file, delimiter=',')


def extract_data(data):
    x = data[:, :-1]
    y = data[:, -1].astype('int')
    return x, y

def get_accuracy(model, x, y):
    return np.sum(model.predict(x) == y) * 100 / y.size


x, y = extract_data(train_data[2:])
x_test, y_test = extract_data(test_data[2:])
x_val, y_val = extract_data(val_data[2:])


# optimal parameters found:
n_esimators_opt = 450
max_features_opt = 0.7
min_samples_split_opt = 2


# vary max_features
mf_range = [0.1, 0.3, 0.7, 0.9]
mf_test_acc = []
mf_val_acc = []

for max_features in mf_range:
    print("mf", max_features)
    rf_clf = RandomForestClassifier(max_depth=40, max_features=max_features, n_estimators=n_esimators_opt, min_samples_split=min_samples_split_opt, oob_score=True, n_jobs=-1, verbose=1)

    rf_clf.fit(x, y)
    mf_test_acc.append(get_accuracy(rf_clf, x_test, y_test))
    mf_val_acc.append(get_accuracy(rf_clf, x_val, y_val))

plt.xlabel("max_features")
plt.ylabel("accuracy")
plt.plot(mf_range, mf_test_acc, label="test accuracy")
plt.plot(mf_range, mf_val_acc, label="val accuracy")
plt.legend()
plt.savefig("max_features.png")


# vary n_estimators
ne_test_acc = []
ne_val_acc = []
ne_range = [2, 10, 50, 200, 400, 600]

for ne in ne_range:
    print(ne)
    rf_clf = RandomForestClassifier(max_depth=30, max_features=max_features_opt, n_estimators=ne, min_samples_split=min_samples_split_opt, oob_score=True, n_jobs=-1, verbose=1)

    rf_clf.fit(x, y)
    ne_test_acc.append(get_accuracy(rf_clf, x_test, y_test))
    ne_val_acc.append(get_accuracy(rf_clf, x_val, y_val))

plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.plot(list(ne_range), ne_test_acc, label="test accuracy")
plt.plot(list(ne_range), ne_val_acc, label="val accuracy")
plt.legend()
plt.savefig("n_estimators_acc.png")


# vary min_samples_split
ms_range = [2, 4, 8, 16, 32]
ms_test_acc = []
ms_val_acc = []

for ms in ms_range:
    print(ms)
    rf_clf = RandomForestClassifier(max_depth=30, max_features=max_features_opt, n_estimators=n_esimators_opt, min_samples_split=ms, oob_score=True, n_jobs=-1, verbose=1)

    rf_clf.fit(x, y)
    ms_test_acc.append(get_accuracy(rf_clf, x_test, y_test))
    ms_val_acc.append(get_accuracy(rf_clf, x_val, y_val))

plt.xlabel("min_samples_split")
plt.ylabel("accuracy")
plt.plot(list(ms_range), ms_test_acc, label="test accuracy")
plt.plot(list(ms_range), ms_val_acc, label="val accuracy")
plt.legend()
plt.savefig("min_samples_split_acc.png")