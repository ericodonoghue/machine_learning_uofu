import numpy as np
import pandas as pd

length = 5

def perceptron(D, test, r, T):
    w = np.array([0]*length)
    for _ in range(1, T+1):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            y_i = 1 if x_i[length] == 1 else -1
            if ((w.T @ x_i[:-1])*y_i <= 0):
                w = w + ((r*y_i)*x_i[:-1])

    predictions = []
    for x_i in test.values:
        p = 1 if w.T @ x_i[:-1] >= 0 else 0
        predictions.append(p)

    return w, np.mean(predictions == test['label'])

def voted_perceptron(D, test, r, T):
    w = []
    w_0 = np.array([0]*length)
    w.append(w_0)
    c,m,n = [0],0,0

    for _ in range(1, T+1):
        for x_i in D.values:
            y_i = 1 if x_i[length] == 1 else -1
            if ((w[m].T @ x_i[:-1])*y_i <= 0):
                w.append(w[m] + ((r*y_i)*x_i[:-1]))
                c.append(n)
                m += 1
                n = 1
            else:
                n += 1

    predictions = []
    for x_i in test.itertuples(index=False):
        sum = 0
        for i in range(len(w)):
            p = 1 if w[i].T @ x_i[:-1] >= 0 else -1
            sum += p*c[i]

        p = 1 if sum >= 0 else 0
        predictions.append(p)

    return w, c, np.mean(predictions == test['label'])

def average_perceptron(D, test, r, T):
    w = np.array([0]*length)
    a = np.array([0]*length)
    for _ in range(1, T+1):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            y_i = 1 if x_i[length] == 1 else -1
            if ((w.T @ x_i[:-1])*y_i <= 0):
                w = w + ((r*y_i)*x_i[:-1])
            a = a + w

    predictions = []
    for x_i in test.values:
        p = 1 if a.T @ x_i[:-1] >= 0 else 0
        predictions.append(p)

    return a, np.mean(predictions == test['label'])

def run_standard_perceptron():
    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r = 0.25
    T = 10
    w, e = perceptron(train, test, r, T)

    print("Standard Perceptron")
    print(f"Learned Weight Vector: {w}") 
    print(f"Learing Rate r: {r}")
    print(f"Average Prediction Error: {1-e}")

def run_voted_perceptron():
    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r = 0.25
    T = 10
    w, c, e = voted_perceptron(train, test, r, T)

    print("Voted Perceptron")
    #print(f"Learned Weight Vector: {w}") 
    print(f"Learing Rate r: {r}")
    print(f"Average Prediction Error: {1-e}")

    print("c, w")
    for i in range(len(w)):
        print(f" {c[i]}: {w[i]}")
    
def run_average_perceptron():
    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r = 0.25
    T = 10
    a, e = average_perceptron(train, test, r, T)

    print("Average Perceptron")
    print(f"Learned Weight Vector: {a}") 
    print(f"Learing Rate r: {r}")
    print(f"Average Prediction Error: {1-e}")

def main():
    run_standard_perceptron()
    run_voted_perceptron()
    run_average_perceptron()

main()