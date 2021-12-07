import numpy as np
import pandas as pd

def logistic_regression_MAP(D, test, v, T, r_0, d):
    w = np.array([0]*(D.shape[1] - 1))
    r = r_0
    for t in range (T):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            y_i = x_i[D.shape[1] - 1]

            # compute gradient of objective function
            s = (w.T @ x_i[:-1]) * y_i
            grad = - D.shape[0] * y_i * x_i[:-1] / (1 + np.exp(s)) + (w / v**2) # see q4 
            w = w - (r * grad)
        r = r_0 / (1 + ((r_0 / d) * t))

        
    train_predictions = []
    for x_i in D.values:
        p = 1 if w.T @ x_i[:-1] > 0 else -1
        train_predictions.append(p)

    test_predictions = []
    for x_i in test.values:
        p = 1 if w.T @ x_i[:-1] > 0 else -1
        test_predictions.append(p)

    return w, np.mean(train_predictions == D['label']), np.mean(test_predictions == test['label'])

def logistic_regression_ML(D, test, v, T, r_0, d):
    w = np.array([0]*(D.shape[1] - 1))
    r = r_0
    for t in range (T):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            y_i = x_i[D.shape[1] - 1]

            # compute gradient of objective function
            s = (w.T @ x_i[:-1]) * y_i
            grad = - D.shape[0] * y_i * x_i[:-1] / (1 + np.exp(s)) # derivative of ML objective function wrt w
            w = w - (r * grad)
        r = r_0 / (1 + ((r_0 / d) * t))
        
    
    train_predictions = []
    for x_i in D.values:
        p = 1 if w.T @ x_i[:-1] > 0 else -1
        train_predictions.append(p)

    test_predictions = []
    for x_i in test.values:
        p = 1 if w.T @ x_i[:-1] > 0 else -1
        test_predictions.append(p)

    return w, np.mean(train_predictions == D['label']), np.mean(test_predictions == test['label'])


def run_logistic_regression():
    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)
    train.label.replace(0, -1, inplace=True)

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)
    test.label.replace(0, -1, inplace=True)

    r_0 = 0.01
    d = 0.1
    T = 100
    v_ = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    print("--------Logistic Regression--------")
    print("MAP Estimation")
    for v in v_:
        w, train_e, test_e = logistic_regression_MAP(train, test, v, T, r_0, d)
        print(f"Prior Variance: {v}") 
        print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
        print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
        print()
    print("ML Estimation")
    w, train_e, test_e = logistic_regression_ML(train, test, v, T, r_0, d)
    print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
    print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
    print()

    return

def main():
    run_logistic_regression()

main()