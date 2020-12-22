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


# grid search over parameters
parameters = {
    'n_estimators': list(np.arange(50, 451, 100)),
    'max_features': list(np.arange(0.1, 1, 0.2)),
    'min_samples_split': list(np.arange(2, 11, 2))
    }

from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier(max_depth=30, oob_score=True, n_jobs=-1, verbose=1)

gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
gs_clf.fit(x, y)

# optimal parameters
print(gs_clf.best_params_)

# optimal parameters found:
n_esimators_opt = 450
max_features_opt = 0.7
min_samples_split_opt = 2

# train with optimal parameters
rf_clf = RandomForestClassifier(n_estimators=n_esimators_opt, min_samples_split=min_samples_split_opt, max_features=max_features_opt, max_depth=30, oob_score=True, n_jobs=-1, verbose=1)
rf_clf.fit(x, y)

# print accuracy
print("train acc", get_accuracy(rf_clf, x, y))
print("test acc", get_accuracy(rf_clf, x_test, y_test))
print("val acc", get_accuracy(rf_clf, x_val, y_val))