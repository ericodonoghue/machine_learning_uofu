import numpy as np
import pandas as pd
import random as r
import math
# Imports all the collumn and attribute data from another file
from data_info import bank_attribute_types, bank_columns, bank_attribute_values, bank_labels

columns = []
labels = []
attribute_values = {}
attribute_types = {}

trained_trees = []


class Node:
    def __init__(self, branch_value=None, split_attribute=None, label=None, median=0):
        self.branch_value = branch_value
        self.children = []
        self.split_attribute = split_attribute
        self.label = label
        self.median = median


def get_entropy(sv, S, column_name):
    e = 0
    if sv.shape[0] == 0: 
        return 0
    for l_val in labels:
        r = sv[sv['label'] == l_val].shape[0]
        if r == 0: continue
        e -= (r / sv.shape[0]) * math.log2(r / sv.shape[0] + 1e-8)
    return e * (sv.shape[0] / S[column_name].shape[0])

# calculates entropy of the system before choosing a column
def get_before_split_entropy(S):
    b = 0
    for p in S['label'].value_counts(normalize=True):
        b -= p * math.log2(p+1e-8)

    return b


def get_after_split_entropy(S, A):
    a = 0
    if attribute_types[A] == "categorical":
        for val in attribute_values[A]:
            sv = S[S[A] == val]
            a += get_entropy(sv, S, A)
    else:
        median = np.median(S[A])
        S_b = S[S[A] <= median]
        S_a = S[S[A] > median]
        a += get_entropy(S_b, S, A)
        a += get_entropy(S_a, S, A)

    return a
        
# Calculates the entire information gain of splitting on a column name
def ig_gain(S, A):
    b = get_before_split_entropy(S)
    a = get_after_split_entropy(S, A)

    return b - a

def get_score(data, column_name):
    return ig_gain(data, column_name)

def best_split_attribute(S, attributes):
    A = None
    max_gain = float('-inf')  
    for attribute in attributes:
        if S[[attribute]].shape[0] > 0:
            s = get_score(S[[attribute,'label']], attribute)
            if s > max_gain:
                max_gain = s
                A = attribute
    return A

# because we are using pandas data frames, S will also contain the the label (target attribute) information
# returns root node of the learned decision tree
def ID3(S, attributes, branch_value, depth, max_depth):
    # Create root node for the current subtree
    root = Node(branch_value=branch_value)

    # base case: all labels are same - return a leaf node (root of current subtree) with the label
    _labels = S['label'].to_numpy()
    if np.all(_labels == _labels[0]):
        root.label = _labels[0]
        return root

    if len(attributes) == 0 or depth >= max_depth:
        root.label = S['label'].mode()[0]
        return root

    A = best_split_attribute(S, attributes)

    # root node for the current subtree will split on the attribute with max gain
    root.split_attribute = A

    # for each possible value v that A can take (eg safety - low,med,high)
    if attribute_types[A] == "categorical":
        for v in attribute_values[A]:
            # because branchs are stored in the child we do not add a branch here
            Sv = S[S[A] == v] # get all of the rows in which A has the value of v

            if Sv.shape[0] == 0:
                root.children.append(Node(branch_value=v, label=S['label'].mode()[0]))
            else:
                root.children.append(ID3(Sv, [a for a in attributes if a != A], v, depth + 1, max_depth))
    # each possible value for a numerical column is either below or above the average - two cases
    else:
        median = np.median(S[A])
        root.median = median
        V = {"below_avg":S[S[A] <= median], "above_avg":S[S[A] > median]}
        for v in V:
            sv = V[v]
            if sv.shape[0] == 0:
                root.children.append(Node(branch_value=v, label=S['label'].mode()[0]))
            else:
                root.children.append(ID3(sv, [a for a in attributes if a != A], v, depth + 1, max_depth))

    return root

# pulled from original id3 implementation, predicts just one row instead of the whole data set
def predict_row(root, row):
    node = root
    while node.split_attribute: # leaves do not have split attributes thus will end on a leaf node
        for child in node.children:
            # for each case below we have found the path to take thus go to the child and break
            # to the next iteration of while loop
            if (child.branch_value == "above_avg"):
                if (node.median < getattr(row, node.split_attribute)):
                    node = child
                    break
            elif (child.branch_value == "below_avg"):
                if (node.median >= getattr(row, node.split_attribute)):
                    node = child
                    break
            elif child.branch_value == getattr(row, node.split_attribute):
                node = child
                break
    return node.label # label in the leaf node found using the stump

def bagged_predict_row(row):
    predictions = []
    for tree in trained_trees:
        predictions.append(predict_row(tree, row))
    return max(set(predictions), key=predictions.count)

def bagged_predict_row(trees, row):
    predictions = []
    for tree in trees:
        predictions.append(predict_row(tree, row))
    return max(set(predictions), key=predictions.count)

def decompostion(train, test):

    bagged_predictors = {}
    for i in range(100):
        sample_data = train.sample(n=1000)
        for _ in range(500):
            bagged_trees(train.sample(n=1000), columns[:-1])
        bagged_predictors[i] = trained_trees.copy()
        trained_trees.clear()
        print(i)

    b = 0
    v = 0
    for row in test.itertuples(index=False):
        p = []
        for i in range(100):
            root = bagged_predictors[i][0]
            p.append(1 if predict_row(root, row) == "yes" else 0)

        avg = np.mean(p)
        gtl = 1 if row.label == "yes" else 0
        b += (gtl-avg)**2

        sum = 0
        for i in range(100):
            sum += ((1 if row.label == 'yes' else 0) - avg)**2
        v += sum/99
    avg_bias_single = b / 100
    avg_var_single = v / 100
    gse_single = avg_bias_single + avg_var_single

    print("Single, Bias, Var, GSE")
    print(f"{avg_bias_single},{avg_var_single},{gse_single}")

    b = 0
    v = 0
    for row in test.itertuples(index=False):
        # Compute prediction of 100 single trees.
        p = []
        for i in range(100):
            p.append(1 if bagged_predict_row(bagged_predictors, row) == "yes" else 0)

        avg = np.mean(p)
        gtl = 1 if row.label == "yes" else 0
        b += (gtl-avg)**2

        sum = 0
        for i in range(100):
            sum += ((1 if row.label == 'yes' else 0) - avg)**2
        v += sum/99
    avg_bias_bagged = b / 100
    avg_var_bagged = v / 100
    gse_bagged = avg_bias_bagged + avg_var_bagged

    print("Bagged, Bias, Var, GSE")
    print(f"{avg_bias_bagged},{avg_var_bagged},{gse_bagged}")


def bagged_trees(S, columns):
    m = S.shape[0]
    samples = []
    for _ in range(m):
        i = r.randint(0, m-1)
        s = S.iloc[[i]]
        samples.append(s)
    XY_t = pd.concat(samples) # attributes and labels stored in the same dataframe

    trained_trees.append(ID3(XY_t, columns, None, 0, float('inf')))


def predict(S):
    r = []
    for row in S.itertuples(index=False):
        p = []
        for t in trained_trees:
            p.append(predict_row(t, row))
        r.append(max(set(p), key=p.count))
    return np.mean(r == S['label'])

def main():
    global columns
    global attribute_values
    global attribute_types
    global labels
    columns = bank_columns
    attribute_values = bank_attribute_values
    attribute_types = bank_attribute_types
    labels = bank_labels

    train = pd.read_csv("bank-1/train.csv", header=None)
    train.columns = columns
    test = pd.read_csv("bank-1/test.csv", header=None)
    test.columns = columns

    print("BAGGED TREES")
    print("T, train, test")
    for T in range(1, 501):
        bagged_trees(train, columns[:-1])
        t1 = "{0:.4f}".format(predict(train))
        t2 = "{0:.4f}".format(predict(test))
        print(f"{T}, {t1}, {t2}")

    decompostion(train, test)


main()
