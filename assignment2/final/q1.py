import json
import numpy as np
import sys
import math
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from functools import lru_cache


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


def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)


def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english')).union(set(punctuation)).union(["n't", "'s"]) # add punctuations and additional stopwords

    ps = PorterStemmer()
    p_stemmer = lru_cache(maxsize=None)(ps.stem)
    if isinstance(docs, list):
        output_docs = []
        doc_count = -1
        for item in docs:
            doc_count += 1
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens)


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

    docs = getStemmedDocuments(docs)
    return y, docs


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def predict(x, r, phi, theta):
    """
        Args:
            x: input point
            r: number of classes
            phi, theta: learned parameters
        Returns:
            Most expected class label
    """
    max_prob = -math.inf
    argmax_k = -1
    for k in range(r):
        n = len(x)
        summation = 0
        for i in range(n):
            l = x[i] - 1
            summation += np.log(theta[l][k])
        prob_k = summation + np.log(phi[k])
        if prob_k > max_prob:
            max_prob = prob_k
            argmax_k = k
    return argmax_k + 1


def run(train_data, test_data):
    """
        Naive Bayes with bigrams
    """

    y, docs = load_data(train_data)
    y_test, docs_test = load_data(test_data)

    # construct x, x_test and vocabulary
    vocabulary = {}
    x = []
    x_test = []

    for doc in docs:
        xi = []
        for w in range(len(doc)-1):
            bg = (doc[w], doc[w+1])
            if bg not in vocabulary:
                vocabulary[bg] = len(vocabulary) + 1
            xi.append(vocabulary[bg])
        x.append(xi)

    for doc in docs_test:
        xi = []
        for w in range(len(doc)-1):
            bg = (doc[w], doc[w+1])
            if bg in vocabulary:
                xi.append(vocabulary[bg])
        x_test.append(xi)

    m = len(y)
    m_test = len(y_test)
    r = 5 # number of classes
    V = len(vocabulary)

    # y takes values in {1, 2, ..., r}; parameterized by phi
    # x takes values in {1, 2, ..., V}; parameterized by theta

    # evaluate phi
    phi = np.zeros(r)
    for i in range(m):
        k = y[i] - 1
        phi[k] += 1
    phi = phi / m

    # evaluate theta
    theta_numerator = np.zeros((V, r))
    theta_denominator = np.zeros((r))
    for i in range(m):
        ni = len(x[i])
        k = y[i] - 1
        for j in range(ni):
            l = x[i][j] - 1
            theta_numerator[l][k] += 1
        theta_denominator[k] += ni

    theta = np.zeros((V, r))
    for j in range(V):
        for k in range(r):
            theta[j][k] = (theta_numerator[j][k] + 1) / (theta_denominator[k] + V)

    # Make predictions
    test_predictions = np.zeros(m_test, np.int)
    for i in range(m_test):
        test_predictions[i] = predict(x_test[i], r, phi, theta)

    return test_predictions


def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
