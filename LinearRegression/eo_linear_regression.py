import numpy as np
import pandas as pd
import math

length = 8




def get_normal(a, b):
    v = [0]*len(a)
    for i in range(len(a)):
        v[i] = (a[i] - b[i]) ** 2

    return math.sqrt(sum(v))

def transpose(x, w):
    result = 0
    for i in range(len(x)):
        result += x[i] * w[i]
    return result

def cost(w, S):
    sum = 0
    for x_i in S.itertuples(index=False):
        sum += (x_i[length] - transpose(w, x_i))**2
    return sum / 2

def compute_gradient(S, w_t):
    gradient = [0]*length #intialize gradient vector
        
    # the gradient is comprised of a sum over every row done over every column
    for j in range(length):  # iterating over columns          
        sum = 0
        for x_i in S.itertuples(index=False): # iterating over every row forming sum for one gradient value
            sum -= (x_i[length] - transpose(w_t, x_i))*x_i[j] # derivative of cost function
        gradient[j] = sum
    
    return gradient



def batch_gradient_descent(S, r):
    norm_diff = float('inf')

    # we need two trackers for w as we must update w_t+1 using w_t
    w_t = [0]*length
    w_t_1 = [0]*length
    t = 0
    
    while norm_diff > 10**(-6):
        # before doing anything else print the cost with the current weight vector
        print(f"{t}: {cost(w_t, S)}") 

        gradient = compute_gradient(S, w_t)

        for j in range(length):
            w_t_1[j] = w_t[j] - r*gradient[j]

        norm_diff = get_normal(w_t_1, w_t)
        w_t = w_t_1.copy()
        t += 1

    return w_t


def run_batch():
    # Get and set the global variables
    train = pd.read_csv("concrete/train.csv", header=None)
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("concrete/test.csv", header=None)
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r = 0.0078125
    w = batch_gradient_descent(train, r)

    print("Batch Gradient Descent")
    print(f"Learned Weight Vector: {w}") 
    print(f"Learing Rate r: {r}")
    print(f"Cost of Test Data: {cost(w, test)}")


def stochastic_gradient_descent(S, r):
    w = [0]*length
    for _ in range(1000):    
        for x_i in S.itertuples(index=False):
            for j in range(length):
                w[j] = w[j] + (r*(x_i[length] - transpose(w, x_i))*x_i[j]) # update w as we go 
    return w


def run_stochastic():
    # Get and set the global variables
    train = pd.read_csv("concrete/train.csv", header=None)
    bias_fold_in = [1]*train.shape[0]
    train.insert(0, "bias", bias_fold_in)

    test = pd.read_csv("concrete/test.csv", header=None)
    bias_fold_in = [1]*test.shape[0]
    test.insert(0, "bias", bias_fold_in)

    r = 0.0078125
    w = stochastic_gradient_descent(train, r)

    print("Stochastic Gradient Descent")
    print(f"Learned Weight Vector: {w}") 
    print(f"Learing Rate r: {r}")
    print(f"Cost of Test Data: {cost(w, test)}")

def main():
    run_batch()
    run_stochastic()


main()
