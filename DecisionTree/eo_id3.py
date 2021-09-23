import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import math
import sys

from pandas.core.indexing import is_nested_tuple

class Node:
    def __init__(self, branch_value=None, split_attribute=None, depth=0, label=None, median=0):
        self.branch_value = branch_value
        self.children = []
        self.split_attribute = split_attribute
        self.label = label
        self.depth = depth
        self.median = median


def ig_gain(S, A):
    before_split_entropy = 0
    for p in S['label'].value_counts(normalize=True):
        before_split_entropy -= p * math.log2(p+1e-8)

    after_split_entropy = 0

    if (attr_type[A] == 'categorical'):
        for a_val in attr_values[A]:
            sv = S[S[A] == a_val]
            sv_entropy = 0
            for l_val in labels:
                if sv.shape[0] == 0: continue
                r = sv[sv['label'] == l_val].shape[0]
                sv_entropy -= (r / sv.shape[0]) * math.log2((r / sv.shape[0]) + 1e-8)

            after_split_entropy += sv_entropy * (sv.shape[0] / S[A].shape[0])
    else:
        median = np.median(S[A])
        Sv_b = S[S[A] <= median]
        Sv_a = S[S[A] > median]
        
        sv_entropy = 0
        for l_val in labels:
            if Sv_b.shape[0] == 0: continue
            r = Sv_b[Sv_b['label'] == l_val].shape[0]
            sv_entropy -= (r / Sv_b.shape[0]) * math.log2((r / Sv_b.shape[0]) + 1e-8)
        after_split_entropy += sv_entropy * (Sv_b.shape[0] / S[A].shape[0])
        
        sv_entropy = 0
        for l_val in labels:
            if Sv_a.shape[0] == 0: continue
            r = Sv_a[Sv_a['label'] == l_val].shape[0]
            sv_entropy -= (r / Sv_a.shape[0]) * math.log2((r / Sv_a.shape[0]) + 1e-8)
        after_split_entropy += sv_entropy * (Sv_a.shape[0] / S[A].shape[0])
            
    return before_split_entropy - after_split_entropy

def me_gain(S, A):
    before_split_me = float('-inf')
    for p in S['label'].value_counts():
        before_split_me = max(p, before_split_me)

    before_split_me = (S['label'].shape[0] - before_split_me) / S['label'].shape[0]

    after_split_me = 0

    if (attr_type[A] == 'categorical'):
        for a_val in attr_values[A]:
            sv = S[S[A] == a_val]
            if (sv.shape[0] <= 0): continue
            sv_me = float('-inf')
            for l_val in labels:
                sv_me = max(sv_me, sv[sv['label'] == l_val].shape[0])
            sv_me = (sv.shape[0] - sv_me) / sv.shape[0]

            after_split_me += sv_me * (sv.shape[0] / S[A].shape[0])
    else:
        median = np.median(S[A])
        Sv_b = S[S[A] <= median]
        Sv_a = S[S[A] > median]
        
        if (Sv_b.shape[0] > 0):
            sv_me = float('-inf')
            for l_val in labels:
                sv_me = max(sv_me, Sv_b[Sv_b['label'] == l_val].shape[0])
            sv_me = (Sv_b.shape[0] - sv_me) / Sv_b.shape[0]

            after_split_me += sv_me * (Sv_b.shape[0] / S[A].shape[0])
        
        if (Sv_a.shape[0] > 0): 
            sv_me = float('-inf')
            for l_val in labels:
                sv_me = max(sv_me, Sv_a[Sv_a['label'] == l_val].shape[0])
            sv_me = (Sv_a.shape[0] - sv_me) / Sv_a.shape[0]

            after_split_me += sv_me * (Sv_a.shape[0] / S[A].shape[0])
            
    return before_split_me - after_split_me

def gi_gain(S, A):
    before_split_gi = 0
    for p in S['label'].value_counts(normalize='True'):
        before_split_gi += p**2

    before_split_gi = 1 - before_split_gi

    after_split_gi = 0
    if (attr_type[A] == 'categorical'):
        for a_val in attr_values[A]:
            sv = S[S[A] == a_val]
            if (sv.shape[0] <= 0): continue
            sv_gi = 0
            for l_val in  labels:
                r = sv[sv['label'] == l_val].shape[0]
                sv_gi += (r / sv.shape[0])**2
            sv_gi = 1 - sv_gi

            after_split_gi += sv_gi * (sv.shape[0] / S[A].shape[0])
    else:
        median = np.median(S[A])
        Sv_b = S[S[A] <= median]
        Sv_a = S[S[A] > median]

        if (Sv_b.shape[0] > 0):
            sv_gi = 0
            for l_val in  labels:
                r = Sv_b[Sv_b['label'] == l_val].shape[0]
                sv_gi += (r / Sv_b.shape[0])**2
            sv_gi = 1 - sv_gi
            after_split_gi += sv_gi * (Sv_b.shape[0] / S[A].shape[0])
        
        if (Sv_a.shape[0] > 0):
            sv_gi = 0
            for l_val in  labels:
                r = Sv_a[Sv_a['label'] == l_val].shape[0]
                sv_gi += (r / Sv_a.shape[0])**2
            sv_gi = 1 - sv_gi
            after_split_gi += sv_gi * (Sv_a.shape[0] / S[A].shape[0]) 
            
    return before_split_gi - after_split_gi


def score(data, column_name):
    if gain_type == 'ME':
        return me_gain(data, column_name)
    elif gain_type == 'GI':
        return gi_gain(data, column_name)
    else:
        return ig_gain(data, column_name)


def best_split_attribute(S, attributes):
    A = None
    max_gain = float('-inf')  
    for attribute in attributes:
        if S[[attribute]].shape[0] > 0:
            s = score(S[[attribute, 'label']], attribute)
            if s > max_gain:
                max_gain = s
                A = attribute
    return A


# because we are using pandas data frames, S will also contain the the label (target attribute) information
def ID3(S, attributes, branch_value, depth):
    # Create root node for the current subtree
    root = Node(branch_value=branch_value, depth=depth+1)

    # base case: all labels are same - return a leaf node (root of current subtree) with the label
    label_arr = S['label'].to_numpy()
    if np.all(label_arr == label_arr[0]):
        root.label = label_arr[0]
        return root

    # base case: attributes is empty - return a leaf node (root of current subtree) with most common label
    if len(attributes) == 0 or root.depth >= max_depth:
        root.label = S['label'].mode()[0]
        return root

    A = best_split_attribute(S, attributes)

    # root node for the current subtree will split on the attribute with max gain
    root.split_attribute = A

    # for each possible value v that A can take (eg saftey - low,med,high)
    if (attr_type[A] == 'categorical'):
        for v in attr_values[A]:
            # because branchs are stored in the child we do not add a branch here

            Sv = S[S[A] == v] # get all of the rows in which A has the value of v

            if Sv.shape[0] == 0:
                root.children.append( Node(branch_value=v, label=S['label'].mode()[0], depth=root.depth+1) ) 
            else:
                root.children.append( ID3(Sv, [a for a in attributes if a != A], v, root.depth) ) # recursive call
    else:
        median = np.median(S[A])
        Sv_b = S[S[A] <= median]
        Sv_a = S[S[A] > median]
        root.median = median
        if Sv_b.shape[0] == 0:
            root.children.append( Node(branch_value="below_avg", label=S['label'].mode()[0], median=median) )
        else:
            root.children.append( ID3(Sv_b, [a for a in attributes if a != A], "below_avg", depth + 1) )
        if Sv_a.shape[0] == 0:
            root.children.append( Node(branch_value="above_avg", label=S['label'].mode()[0], median=median) )
        else:
            root.children.append( ID3(Sv_a, [a for a in attributes if a != A], "above_avg", depth + 1) )

    
    return root


def Predict(root_node, test):
    results = []
    i = 1
    for row in test.itertuples(index=False):
        node = root_node
        while node.split_attribute:
            for child in node.children:
                if (child.branch_value == "below_avg"):
                    if (node.median > getattr(row, node.split_attribute)):
                        node = child
                        break
                elif (child.branch_value == "above_avg"):
                    if (node.median <= getattr(row, node.split_attribute)):    
                       node = child
                    break    
                elif child.branch_value == getattr(row, node.split_attribute):
                    node = child
                    break

        results.append(node.label)
        i += 1

    return results


def generate_report_car():

    global labels
    global columns
    global attr_values
    global attr_type
    global max_depth
    global gain_type
    

    labels = ['unacc', 'acc', 'good', 'vgood']
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    attr_values = {   'buying' : ['vhigh', 'high', 'med', 'low']
                    , 'maint' : ['vhigh', 'high', 'med', 'low']
                    , 'doors' : ['2', '3', '4', '5more']
                    , 'persons' : ['2', '4', 'more']
                    , 'lug_boot' : ['small', 'med', 'big']
                    , 'safety' : ['low', 'med', 'high']}
    attr_type = { 'buying': "categorical",
                  'maint': "categorical",
                  'doors': "categorical",
                  'persons': "categorical",
                  'lug_boot': "categorical",
                  'safety': "categorical",}

    print("CARS")
    # read in training data into a pandas data frame and set col names
    train = pd.read_csv('car/train.csv')
    train.columns = columns

    # read in testing data into a pandas data frame and set col names
    test = pd.read_csv('car/test.csv')
    test.columns = columns

    for h in ['IG','ME','GI']:
        for d in range(1,7):
            gain_type = h
            max_depth = d
            id3_tree = ID3(train, columns[:-1], None, -1)

            predictions = Predict(id3_tree, test)
            p = round(np.mean(predictions == test['label']), 4)
            print(f"test: {p} {h} {d}")

            predictions = Predict(id3_tree, train)
            p = round(np.mean(predictions == train['label']), 4)
            print(f"train: {p} {h} {d}")
            print()

def generate_report_bank_no_missing():

    global labels
    global columns
    global attr_values
    global attr_type
    global max_depth
    global gain_type
    

    labels = ['no', 'yes']
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    attr_values = { 'age': "numeric",
                  'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
                  'marital': ["married","divorced","single"],
                  'education': ["unknown","secondary","primary","tertiary"],
                  'default': ["yes","no"],
                  'balance': "numeric",
                  'housing': ["yes","no"],
                  'loan': ["yes","no"],
                  'contact': ["unknown","telephone","cellular"],
                  'day': "numeric",
                  'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                  'duration': "numeric",
                  'campaign': "numeric",
                  'pdays': "numeric",
                  'previous': "numeric",
                  'poutcome': ["unknown","other","failure","success"]}
    attr_type = { 'age': "numeric",
                  'job': "categorical",
                  'marital': "categorical",
                  'education': "categorical",
                  'default': "categorical",
                  'balance': "numeric",
                  'housing': "categorical",
                  'loan': "categorical",
                  'contact': "categorical",
                  'day': "numeric",
                  'month': "categorical",
                  'duration': "numeric",
                  'campaign': "numeric",
                  'pdays': "numeric",
                  'previous': "numeric",
                  'poutcome': "categorical"}

    print("BANK treat unknown as not missing")
    # read in training data into a pandas data frame and set col names
    train = pd.read_csv('bank/train.csv')
    train.columns = columns

    # read in testing data into a pandas data frame and set col names
    test = pd.read_csv('bank/test.csv')
    test.columns = columns

    # predict using unknown as not missing
    for h in ['IG','ME','GI']:
        for d in range(1,17):
            gain_type = h
            max_depth = d
            id3_tree = ID3(train, columns[:-1], None, -1)

            predictions = Predict(id3_tree, test)
            p = round(np.mean(predictions == test['label']), 4)
            print(f"test: {p} {h} {d}")

            predictions = Predict(id3_tree, train)
            p = round(np.mean(predictions == train['label']), 4)
            print(f"train: {p} {h} {d}")
            print()


def replace_unknowns(column_name, train):
    c = train[column_name].value_counts().sort_values(ascending=False).index.tolist()
    if (c[0] == "unknown"): c[0] = c[1]
    train.replace({column_name: 'unknown'}, c[0], inplace=True)

def generate_report_bank_missing_most_common():

    global labels
    global columns
    global attr_values
    global attr_type
    global max_depth
    global gain_type
    

    labels = ['no', 'yes']
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    attr_values = { 'age': "numeric",
                  'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
                  'marital': ["married","divorced","single"],
                  'education': ["unknown","secondary","primary","tertiary"],
                  'default': ["yes","no"],
                  'balance': "numeric",
                  'housing': ["yes","no"],
                  'loan': ["yes","no"],
                  'contact': ["unknown","telephone","cellular"],
                  'day': "numeric",
                  'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                  'duration': "numeric",
                  'campaign': "numeric",
                  'pdays': "numeric",
                  'previous': "numeric",
                  'poutcome': ["unknown","other","failure","success"]}
    attr_type = { 'age': "numeric",
                  'job': "categorical",
                  'marital': "categorical",
                  'education': "categorical",
                  'default': "categorical",
                  'balance': "numeric",
                  'housing': "categorical",
                  'loan': "categorical",
                  'contact': "categorical",
                  'day': "numeric",
                  'month': "categorical",
                  'duration': "numeric",
                  'campaign': "numeric",
                  'pdays': "numeric",
                  'previous': "numeric",
                  'poutcome': "categorical"}

    print("BANK fill in unknown with most common value")
    # read in training data into a pandas data frame and set col names
    train = pd.read_csv('bank/train.csv')
    train.columns = columns

    # read in testing data into a pandas data frame and set col names
    test = pd.read_csv('bank/test.csv')
    test.columns = columns

    # replace values of unknown in columns that can be unknown
    for column in ["job", "education", "contact", "poutcome"]:
        replace_unknowns(column, train)

    for h in ['IG','ME','GI']:
        for d in range(1,17):
            gain_type = h
            max_depth = d
            id3_tree = ID3(train, columns[:-1], None, -1)

            predictions = Predict(id3_tree, test)
            p = round(np.mean(predictions == test['label']), 4)
            print(f"test: {p} {h} {d}")

            predictions = Predict(id3_tree, train)
            p = round(np.mean(predictions == train['label']), 4)
            print(f"train: {p} {h} {d}")
            print()


generate_report_car()
generate_report_bank_no_missing()
generate_report_bank_missing_most_common()

sys.exit(0)