import numpy as np
import json
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def json_reader(fname):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    for line in open(fname, mode="r"):
        yield json.loads(line)


def load_data(data):
    """
        Returns:
            y: class labels i.e. stars for reviews
            docs: review text
    """
    y = []
    docs = []

    for review in json_reader(data):
        y.append(int(review["stars"]))
        docs.append(review["text"])

    y = np.asarray(y)
    return y, docs


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(train_data, test_data):

    # load train and test data
    y_train, docs = load_data(train_data)
    _, docs_test = load_data(test_data)

    # SVM with LIBLINEAR
    svm_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC()),
    ])

    # Search space for parameters
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__C': (1e-2, 1, 5),
    }

    # 5-fold cross validation over a subset of training data
    gs_clf = GridSearchCV(svm_clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(docs[:int(len(docs)/5)], y_train[:int(len(docs)/5)])

    # optimal parmaters
    vect__ngram_range = gs_clf.best_params_['vect__ngram_range']
    clf__C = gs_clf.best_params_['clf__C']

    # Train on full data with optimal parameters
    svm_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=vect__ngram_range)),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC(C=clf__C)),
    ])

    svm_clf.fit(docs, y_train)

    # Make predictions
    predicted = np.asarray(svm_clf.predict(docs_test), np.int)
    return predicted


def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
