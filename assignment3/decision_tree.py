import sys
import os
import numpy as np


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def extract_data(data):
    x = data[:, :-1]
    y = data[:, -1].astype('int')
    return x, y


def entropy(y, num_labels):
    counts = np.bincount(y, minlength=num_labels+1) # count occurences of each label
    prob = counts / len(y) # probability for each label
    summation = 0
    for k in range(1, num_labels + 1):
        if prob[k] != 0:
            summation += prob[k] * np.log2(prob[k])
    return -summation


def choose_best_attribute(D, num_attributes, num_labels):
    x, y = extract_data(D)
    max_MI = -1
    max_j = -1
    for j in range(num_attributes):
        if j > 10:
            # discrete attribute
            D0 = D[x[:, j] == 0]
            D1 = D[x[:, j] == 1]
        else:
            # continuous attribute 
            median = np.median(x[:, j])
            D0 = D[x[:, j] <= median]
            D1 = D[x[:, j] > median]
        x0, y0 = extract_data(D0)
        x1, y1 = extract_data(D1)

        if (y0.size == 0) or (y1.size == 0): # if there is no data with this attr, then skip this
            continue

        p0 = y0.size / y.size
        p1 = y1.size / y.size

        MI = entropy(y, num_labels) - (entropy(y0, num_labels) * p0 + entropy(y1, num_labels) * p1) # mutual information
        if MI > max_MI:
            max_MI = MI
            max_j = j
    return max_j


all_curr_nodes = []

def dfs_helper(node, visited):
    global all_curr_nodes
    all_curr_nodes.append(node)
    
    visited.add(node)

    for child in node.children:
        if child:
            if child not in visited:
                dfs_helper(child, visited)

def dfs(node):
    global all_curr_nodes
    all_curr_nodes = []
    visited = set()
    dfs_helper(node, visited)



class Node:
    def __init__(self, attribute, majority_class):
        self.attribute = attribute
        self.majority_class = majority_class
        self.children = []

    def is_leaf(self):
        return self.children == None or self.children == []

    def add_child(self, node):
        if self.children == None:
            self.children = [node]
        else:
            self.children.append(node)

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_attribute_median(self, attribute_median):
        self.attribute_median = attribute_median


def create_leaf(class_label):
    leaf = Node(attribute=None, majority_class=class_label)
    return leaf


class DecisionTree:

    def __init__(self, train_data, num_attributes, num_labels, max_depth=99999999):
        self.train_data = train_data
        self.max_depth = max_depth
        self.num_nodes = 0
        self.depth = 0
        self.num_attributes = num_attributes
        self.num_labels = num_labels
        self.node_list = []
        self.nodes_pruned = 0

    def grow_tree(self, D, max_depth):
        x, y = extract_data(D)
        if np.all(y == y[0]):
            # leaf node
            leaf = create_leaf(y[0])
            self.num_nodes += 1
            self.node_list.append(leaf)
            return leaf
        else:
            majority_class = np.argmax(np.bincount(y)[1:])
            j = choose_best_attribute(D, self.num_attributes, self.num_labels)
            
            if j == -1: # if no suitable attr to split is available, then create leaf with majority class
                return create_leaf(majority_class)
            
            node = Node(j, majority_class)
            self.node_list.append(node)
            self.num_nodes += 1
            if j > 10:
                # discrete attribute
                D0 = D[x[:, j] == 0]
                D1 = D[x[:, j] == 1]
            else:
                # continuous attribute 
                attribute_median = np.median(x[:, j])
                D0 = D[x[:, j] <= attribute_median]
                D1 = D[x[:, j] > attribute_median]
                node.set_attribute_median(attribute_median)

            max_depth -= 1
            self.depth = max(self.depth, self.max_depth - max_depth) # update tree depth
            if max_depth == 0:
                return node
            node.add_child(self.grow_tree(D0, max_depth))
            node.add_child(self.grow_tree(D1, max_depth))
            return node

    def train(self):
        self.root = self.grow_tree(self.train_data, self.max_depth)

    def predict(self, x):
        return self.predict_node(self.root, x)

    def predict_node(self, node, x):
        if node.is_leaf():
            return node.majority_class
        else:
            j = node.attribute
            if j > 10:
                # discrete attribute
                return self.predict_node(node.children[int(x[j])], x)
            else:
                # continuous attibute
                if x[j] <= node.attribute_median:
                    child = node.children[0]
                else:
                    child = node.children[1]

                if not child:
                    return node.majority_class
                else:
                    return self.predict_node(child, x)

    def post_prune(self, x_val, y_val):
        # prunning

        while(True):
            max_val_acc_increase = 0
            prune_index = -1

            dfs(self.root)

            for node_index in range(len(all_curr_nodes)):
                node = all_curr_nodes[node_index]
                if not node.is_leaf():
                    val_acc_before = sum(int(self.predict(x_val[i]) == y_val[i]) for i in range(y_val.size)) * 100 / y_val.size

                    node_subtree = node.children
                    node.children = []

                    val_acc_after = sum(int(self.predict(x_val[i]) == y_val[i]) for i in range(y_val.size)) * 100 / y_val.size

                    val_acc_increase = val_acc_after - val_acc_before
                    if val_acc_increase > max_val_acc_increase:
                        max_val_acc_increase = val_acc_increase
                        prune_index = node_index
                    
                    node.children = node_subtree

            if prune_index == -1:
                # stop if the tree is not anymore prunable
                break

            prune_node = all_curr_nodes[prune_index]
            prune_node.children = []
            self.nodes_pruned += 1





def run(train_data, test_data, val_data, question):

    train_data = np.genfromtxt(train_data, delimiter=',')
    test_data = np.genfromtxt(test_data, delimiter=',')
    val_data = np.genfromtxt(val_data, delimiter=',')

    train_data = train_data[2:]
    x, y = extract_data(train_data)
    num_attributes = x.shape[1]
    num_labels = 7

    if question == 1:
        max_depth = 50
    else:
        max_depth = 30

    dtree = DecisionTree(train_data, num_attributes, num_labels, max_depth)
    dtree.train()

    # if question == 2:
    #     x_val, y_val = extract_data(val_data[:2])
    #     dtree.post_prune(x_val, y_val)

    x_test, y_test = extract_data(test_data[2:])
    predictions = [1]
    acc = 0
    for i in range(x_test.shape[0]):
        predictions.append(int(dtree.predict(x_test[i])))
        # acc += int(dtree.predict(x_test[i]) == y_test[i])

    return predictions


def main():
    question = sys.argv[1]
    train_data = sys.argv[2]
    val_data = sys.argv[3]
    test_data = sys.argv[4]
    output_file = sys.argv[5]
    output = run(train_data, test_data, val_data, int(question))
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
