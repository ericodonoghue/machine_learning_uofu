import numpy as np
import pandas as pd
from scipy.optimize import minimize

length = 5


def get_r(r_0, a, t, r_type):
    if (r_type):
        return (r_0) / (1 + ((r_0 / a) * t))
    else:
        return (r_0) / (1 + t)

def k(x_i, x_j, gamma):
    x = x_i - x_j
    return np.exp(-np.sum(np.square(x)) / gamma)

def primal_svm(D, test, r_0, a, T, C, N, r_type):
    w = np.array([0]*length)
    r = r_0
    for t in range(1, T+1):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            y_i = 1 if x_i[length] == 1 else -1

            # compute gradient
            grad = np.copy(w)           
            if ( max(0, 1 - (y_i * (w.T @ x_i[:-1]))) == 0 ):
                grad[0] = 0
            else:
                grad[0] = 0
                grad = grad - ((C*N*y_i)*x_i[:-1])   

            w = w - (r * grad)      

        r = get_r(r_0, a, t, r_type)

    train_predictions = []
    for x_i in D.values:
        p = 1 if w.T @ x_i[:-1] >= 0 else 0
        train_predictions.append(p)

    test_predictions = []
    for x_i in test.values:
        p = 1 if w.T @ x_i[:-1] >= 0 else 0
        test_predictions.append(p)

    return w, np.mean(train_predictions == D['label']), np.mean(test_predictions == test['label'])


def dual_svm(D, test, C):
    x = D.drop(columns=['label']).values
    y = D['label'].values
    x_test = test.drop(columns=['label']).values
    y_test = test['label'].values

    def objective_function(a):  
        sum = 0
        for i in range(0,D.shape[0]):
            _y = y[i] * y
            _a = a[i] * a
            _x = x[i] @ x.T
            sum += np.sum(_x * _y * _a)    

        return (0.5*sum) - a.sum()

    N = D.shape[0]
    a0 = np.random.uniform(low=0, high=C, size=N)
    cons = {'type':'eq','fun': lambda alpha: np.sum(alpha*y), 'jac': lambda alpha: y}
    bnds = [(0,C)] * N

    res = minimize(objective_function, a0, constraints=cons,method='SLSQP',bounds=bnds)
    
    res.x[res.x < 0.01] = 0  
    a = res.x

    #print(np.around(a,4))

    w = np.zeros(x.shape[1])
    for i in range(a.shape[0]):
        w += a[i] * y[i] * x[i]

    b_sum = 0
    for i in range(a.shape[0]):
        b_sum += y[i] - w.T @ x[i]
    b = b_sum / a.shape[0]

    train_predictions = [np.sign(w.T @ x_i + b) for x_i in x]
    test_predictions = [np.sign(w.T @ x_i + b) for x_i in x_test]

    return w, b, np.mean(train_predictions == y), np.mean(test_predictions == y_test)


def dual_kernel_svm(D, test, C, gamma):
    x = D.drop(columns=['label']).values
    y = D['label'].values

    k_mat = w = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k_mat[i][j] = k(x[i], x[j], gamma)

    def objective_function(a):  
        sum = 0
        for i in range(a.shape[0]):
            _y = y[i] * y
            _a = a[i] * a
            _x = k_mat[i] @ k_mat.T
            sum += np.sum(_x * _y * _a)    

        return (0.5*sum) - a.sum()

    N = D.shape[0]
    a0 = np.random.uniform(low=0, high=C, size=N)
    cons = {'type':'eq','fun': lambda alpha: np.sum(alpha*y), 'jac': lambda alpha: y}
    bnds = [(0,C)] * N

    res = minimize(objective_function, a0, constraints=cons,method='SLSQP',bounds=bnds)
    
    res.x[res.x < 0.001] = 0  
    res.x[res.x > C - 0.001] = C
    a = res.x

    w = np.zeros(x.shape[1])
    for i in range(a.shape[0]):
        w += a[i] * y[i] * x[i]

    b_sum = 0
    for i in range(a.shape[0]):
        b_sum += y[i] - w.T @ x[i]
    b = b_sum / a.shape[0]

    train_p = []
    for i in range(x.shape[0]):
        pred = 0
        for j in range(a.shape[0]):
            pred += a[j] * y[i] * k_mat[i][j]
        train_p.append(np.sign(pred))

    x_test = test.drop(columns=['label']).values
    y_test = test['label'].values

    test_p = []
    for i in range(x_test.shape[0]):
        pred = 0
        for j in range(a.shape[0]):
            pred += a[j] * y_test[i] * k_mat[i][j]
        test_p.append(np.sign(pred))

    return a, w, b, np.mean(train_p == y), np.mean(test_p == y_test)

def run_primal_svm():
    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r_0 = 0.0001
    a = 0.5
    T = 100
    C = [0.1145, 0.5727,0.8018]
    N = train.shape[0]
    r_type = [True, False]   

    print("--------Primal SVM--------")
    for r_t in r_type:
        if (r_t): 
            print("Learning Rate Schedule: gamma_0 / 1 + (gamma_0 / a)t") 
            print(f"Initial Gamma: {r_0} a: {a}")
        else: 
            print(f"Learning Rate Schedule: gamma_0 / 1 + t") 
            print(f"Initial Gamma: {r_0}")
        for c in C:
            w, train_e, test_e = primal_svm(train, test, r_0, a, T, c, N, r_t)
            print(f"Trade-Off Value: {c}")
            print(f"Learned Weight Vector: {np.around(w,4)}") 
            print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
            print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
            print()
        print()

    return


def run_dual_svm():

    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    train.loc[train['label'] == 0] = -1

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    test.loc[test['label'] == 0] = -1

    C = [0.1145, 0.5727,0.8018]

    print("--------Dual SVM--------")
    for c in C:
        w, b, train_e, test_e = dual_svm(train,test,c)
        print(f"Trade-Off Value: {c}")
        print(f"Learned Weight Vector: {np.around(w,4)}") 
        print(f"Learned Bias: {np.around(b,4)}") 
        print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
        print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
        print()
    return

def run_dual_kernel_svm():

    columns = ['variance','skewness','curtosis','entropy','label']

    train = pd.read_csv("bank-note/train.csv", header=None)
    train.columns = columns
    train.loc[train['label'] == 0] = -1

    test = pd.read_csv("bank-note/test.csv", header=None)
    test.columns = columns
    test.loc[test['label'] == 0] = -1

    C = [0.1145, 0.5727,0.8018]
    gamma = [0.1,0.5,1,5,100]

    print("--------Dual Kernel SVM--------")
    for c in C:
        print(f"Trade-Off Value: {c}")
        _a = []
        for g in gamma:
            a, w, b, train_e, test_e = dual_kernel_svm(train,test,c,g)
            print(f"Gamma Value: {g}")
            print(f"Number of support vectors: {len([i for i in a if i != 0])}")

            if c == 0.5727 and g != 0.1:
                count = 0
                for i in range(a.shape[0]):
                    if(a[i] != 0 and _a[i] != 0):
                        count += 1
                print(f"Number of overlapping support vectors: {count}")

            _a = a.copy()
            #print(f"Learned Weight Vector: {np.around(w,4)}") 
            #print(f"Learned Bias: {np.around(b,4)}") 
            print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
            print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
            print()
            
    return

def main():
    run_primal_svm()
    run_dual_svm()
    run_dual_kernel_svm()

main()