#!/usr/bin/env python3

'''
    Author: Stiven LaVrenov
    Program: id3.py
    Description: Build an ID3 decision tree and predict based on the created tree
    Complete Usage: ./id3.py [other stuff] [other stuff]
'''

import numpy as np
import sys, math

# take 2 command line args [string: input training data filename] [string: input validation data filename]
if len(sys.argv) != 3:
    print('\n', f'Usage: {sys.argv[0]} [training data] [validation data]', '\n')
    sys.exit(1)

# Node class to store node information
class Node:
    def __init__(self, data, columns, attr):
        self.data = data
        self.columns = columns
        self.attr = attr
        self.s_attr = None
        self.split = None
        self.left = None
        self.right = None
        self.class_ = None
        self.terminal = False

    # Recursively create either intermediate or terminal nodes
    def create_node(self, target):
        train = self.data

        max_gain, max_attr, max_split = max_information_gain(train, self.columns, self.attr, target)
        # print(max_split)

        # If the node is NOT a terminal node
        if max_gain > 0:
            self.s_attr = max_attr
            self.split = max_split

            # Create lhs and rhs dataset splits
            for row in train:
                lhs = np.array([row for row in train if row[max_attr] < max_split])
                rhs = np.array([row for row in train if row[max_attr] >= max_split])
            
            # Get the values within the lhs and rhs datasets respectively
            lhs_attr = np.array([attribute for attribute in lhs])
            rhs_attr = np.array([attribute for attribute in rhs])

            # Create left and right children nodes
            self.left = Node(lhs, self.columns, lhs_attr)
            self.right = Node(rhs, self.columns, rhs_attr)

            # Recursive call for left and right child nodes
            self.left.create_node(target)
            self.right.create_node(target)

        # If the node is a terminal node
        else:
            self.terminal = True

        if self.terminal:
            class_count = np.unique([c[-1] for c in train])
            # PURE CLASS CHECK
            if len(class_count) == 1:
                self.class_ = class_count[0]
            # NOT A PURE CLASS
            else:
                train = train[train[:,-1].argsort()]
                class_col, class_count = np.unique(train[-1], return_counts=True)
                max_count = class_count.argmax()
                self.class_ = class_col[max_count]

    # Recursively predict on the ID3
    def predict(self, data):
        if self.terminal:
            return self.class_
        else:
            if data[self.s_attr] < self.split:
                return self.left.predict(data)
            else:
                return self.right.predict(data)

# ID3 class to initialize and make the decision tree
class ID3:
    def __init__(self):
        self.root = None

    def make_tree(self, data, columns, attr, target):
        self.root = Node(data, columns, attr)
        self.root.create_node(target)

    def predict(self, data):
        return self.root.predict(data)

# Returns the entropy of a given dataset, whether train or a lhs or rhs dataset
def calculate_entropy(data, target):
    entropy = 0

    for c in target:
        elem_count = len([x for x in data if x[-1] == c])
        row_count = len(data)
        probability = elem_count / row_count
        if probability > 0:
            entropy += -probability * math.log(probability, 2)

    return entropy

# Determine the information gain with the lhs and rhs of the dataset
def information_gain(data, binary_col, binary_attr, target):
    lhs = [x for x in data if x[binary_col] < binary_attr]
    rhs = [x for x in data if x[binary_col] >= binary_attr]
    row_count = len(data)

    lhs_probability = len(lhs) / row_count
    rhs_probability = len(rhs) / row_count

    return calculate_entropy(data, target) - (lhs_probability * calculate_entropy(lhs, target) - rhs_probability * calculate_entropy(rhs, target))

# Return the maximum gain, value, and split value in the dataset to split on
def max_information_gain(data, columns, attributes, target):
    max_info_gain = -1
    max_info_attr = None
    max_info_binary_split = None

    # Iterate over each attribute
    for col in columns:
        # Get unique values within the attribute
        attrs = np.unique(data[:,col])
        # Determine each split within the attribute
        splits = binary_splits(attrs)
        for split in splits:
            # Determine the information gain at each split point and keep track of the maximum gain
            split_gain = information_gain(data, col, split, target)

            if split_gain > max_info_gain:
                max_info_gain = split_gain
                max_info_attr = col
                max_info_binary_split = split
    # print(max_info_gain, max_info_attr, max_info_binary_split)
    return max_info_gain, max_info_attr, max_info_binary_split

# Return the binary split points of a given attribute
def binary_splits(attributes):
    averages = lambda x, y : (x + y) / 2
    splits = [averages(attributes[x], attributes[x+1]) for x in range(len(attributes) - 1)]

    return splits

def main():
    # read in training data // one example per line
    train = np.loadtxt(sys.argv[1])
    if len(train.shape) < 2:
        train = np.array([train])

    # read in test data // one example per line
    test = np.loadtxt(sys.argv[2])
    if len(test.shape) < 2:
        test = np.array([test])

    # Get unique labels in the train and test dataset in order to cover each class label
    train_labels = np.unique(train[:,-1])
    test_labels = np.unique(test[:,-1])
    target = list(set(train_labels) | set(test_labels))

    # Get each column index of the dataset and each unique value in the dataset
    columns = list(range(train.shape[1]-1))
    attributes = [attr for attr in train]
    attributes = np.array(attributes)

    # Initialize the ID3 tree
    id3_tree = ID3()
    # Create the ID3 tree
    id3_tree.make_tree(train, columns, attributes, target)

    # Evaluate each correct prediction on the test data
    correct = 0
    for row in test:
        if id3_tree.predict(row) == row[-1]:
            correct += 1
    print(correct)

if __name__ == '__main__':
    main()