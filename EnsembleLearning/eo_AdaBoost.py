import numpy as np
import pandas as pd
import math
# Imports all the collumn and attribute data from another file
from data_info import bank_attribute_types, bank_columns, bank_attribute_values, bank_labels

columns = []
labels = []
attribute_values = {}
attribute_types = {}

A_t = []
H_t = []


class Node:
    def __init__(self, branch_value=None, split_attribute=None, label=None, median=0):
        self.branch_value = branch_value
        self.children = []
        self.split_attribute = split_attribute
        self.label = label
        self.median = median

# Calculates the entropy of a single subset taking weights into account
def get_entropy(sv, S):
    e = 0
    if sv.shape[0] == 0:
        return 0
    for label in labels:
        p = sum(sv[sv['label'] == label]["score"])/sum(sv["score"]) 
        if p == 0: continue
        e -= (p) * math.log2(p)
    return e * (sum(sv["score"]) / sum(S["score"]))

# calculates entropy of the system before choosing a column
def get_before_split_entropy(S):
    b = 0

    # we are now calculating information gain using the weights of each label 
    p_yes = sum(S[S['label'] == "yes"]['score'])
    p_no = sum(S[S['label'] == "no"]['score'])

    # entropy needs to be determined by the sum of weights for each label
    b -= p_yes * math.log2(p_yes+1e-8)
    b -= p_no * math.log2(p_no+1e-8)
    return b

def get_after_split_entropy(S, A):
    a = 0
    if len(attribute_values[A]) != 0:
        for val in attribute_values[A]:
            sv = S[S[A] == val]
            a += get_entropy(sv, S)
    else:
        median = np.median(S[A])
        S_b = S[S[A] <= median]
        S_a = S[S[A] > median]
        a += get_entropy(S_b, S)
        a += get_entropy(S_a, S)
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
            s = get_score(S[[attribute, 'score', 'label']], attribute)
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

    # base case: attributes is empty - return a leaf node (root of current subtree) with most common label 
    # for this use the most common label is the one with the larger sum of weights
    if len(attributes) == 0 or depth >= max_depth:
        root.label = "yes" if sum(S[S['label'] == "yes"]['score']) > sum(S[S['label'] == "no"]['score']) else "no"
        return root

    A = best_split_attribute(S, attributes)

    # root node for the current subtree will split on the attribute with max gain
    root.split_attribute = A

    # for each possible value v that A can take (eg safety - low,med,high)
    if len(attribute_values[A]) != 0:
        for v in attribute_values[A]:
            # because branchs are stored in the child we do not add a branch here
            Sv = S[S[A] == v] # get all of the rows in which A has the value of v

            if Sv.shape[0] == 0:
                root.children.append(Node(branch_value=v, label=("yes" if sum(S[S['label'] == "yes"]['score']) > sum(S[S['label'] == "no"]['score']) else "no")))
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
                root.children.append(Node(branch_value=v, label=("yes" if sum(S[S['label'] == "yes"]['score']) > sum(S[S['label'] == "no"]['score']) else "no")))
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

def get_vote(S, h_t):
    e_t = 0.0
    for x in S.itertuples(index=False):
        h_x_i = predict_row(h_t, x)
        y_i = x.label
        if(h_x_i != y_i):
            e_t += x.score

    return 0.5 * np.log((1-e_t)/e_t)

def update_weights(S, h_t, a_t):
    i = 0 #index used for data frame score replacement
    for D_t in S.itertuples(index=False):
        h_x_i = predict_row(h_t, D_t)
        y_i = D_t.label
        v = math.exp(a_t)
        if(h_x_i == y_i):
            v = math.exp(-a_t)
        S.at[i, 'score'] = D_t.score*v # D_t(i)*e^(a)
        i += 1

    i = 0
    z_t = float(sum(S.score))
    for D_t in S.itertuples(index=False):
        S.at[i, 'score'] = D_t.score/z_t # this is the updated score being normalzied
        i += 1

# due to pandas the lables are contained within S
def ada_boost(S, A, T):
    global H_t, A_t
    H_t = [0]*T
    A_t = [0]*T

    # adaboost step 1
    D_1 = [1/float(S.shape[0])]*S.shape[0]
    S["score"] = D_1

    # adaboost step 2
    for t in range(1, T+1):
        h_t = ID3(S, A, None, 0, 1) # adaboost step 2.1
        a_t = get_vote(S, h_t) # adaboost step 2.2
        H_t[t-1] = h_t
        A_t[t-1] = a_t  
        update_weights(S, h_t, a_t) # adaboost step 2.3
        
        
# adaboost step 3
def get_final_hypothesis(S):
    results = []
    #for each row in the data (S) construct the sum based on how many stumps there were
    for row in S.itertuples(index=False):
        sum = 0
        # for each stump we have build the sum based off the stump and its vote
        for i in range(len(H_t)):
            # for the current stump see if it predicts the row correct
            prediction = predict_row(H_t[i], row)
            H_t_i = -1
            if prediction == "yes": H_t_i = 1
            sum += A_t[i]*H_t_i
        # our prediction
        if(sum > 0):
            results.append("yes")
        else:
            results.append("no")

    return np.mean(results == S['label'])

# Print a report with all the given information
def main():
    # Get and set the global variables
    global columns, attribute_values, attribute_types, labels
    columns = bank_columns
    attribute_values = bank_attribute_values
    attribute_types = bank_attribute_types
    labels = bank_labels

    train = pd.read_csv("bank-1/train.csv", header=None)
    train.columns = columns

    test = pd.read_csv("bank-1/test.csv", header=None)
    test.columns = columns

    print("ADA BOOST")
    print("T, train, test")
    for T in range(1, 501, 5):
        ada_boost(train, columns[:-1], T)
        t1 = "{0:.4f}".format(get_final_hypothesis(train))
        t2 = "{0:.4f}".format(get_final_hypothesis(test))
        print(f"{T}, {t1}, {t2}")


main()
