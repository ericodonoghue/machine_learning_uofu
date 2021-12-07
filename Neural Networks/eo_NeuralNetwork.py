import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import torch
import torch.nn as nn

 #region neural network 

def forward(input, z, w):
    z[0] = input
    for i in range(1, 3):
        w_i = w[i] # get the weights for the current layer 
        z_i = z[i - 1].reshape([-1,1])
        s = w_i @ z_i # see q2 for calculation
        z[i][:-1,:] = sigmoid(s) # sigmoid function for calculating z current z value

    # mutliply the weights right below the output node with the nodes at layer 2 for the y calculation (see q2/q3 written)
    z[3] = w[3] @ z[3 - 1] 

    return z[-1] # output is at last z value (y node)

def back(y, width, z, w, dw):
    # explicit back propagation for each layer in the neural network structure from layers 3-1 (only 3 layers of weights)

    # top node layer (y output)
    layer = -1 # aka the top layer
    dL_dz = z[layer] - y # see q2
    nb_width = 1 # nb - no bias
    z_input = z[layer - 1] # z values from layer one down
    dz_dw =  np.transpose(np.tile(z_input, [1, nb_width]))
    dw[layer] = dL_dz * dz_dw
    dz_dz = w[layer][:, :-1] # for caching

    # hidden layer 
    layer = 2
    nb_width = width - 1
    z_input = z[layer - 1]
    ds_dw = np.transpose(np.tile(z_input, [1, nb_width])) # allows for easy calculation 
    z_output = z[layer][:-1]
    dz_dw = z_output * (1 - z_output) * ds_dw # see q3 this is the derivative of the sigmoid using chain rule
    dL_dz = dz_dz.T @ dL_dz
    dL_dw = dL_dz * dz_dw
    dw[layer] = dL_dw

    dz_dz = z_output * (1 - z_output) * w[layer] #see q3
    dz_dz = dz_dz[:, :-1] # compute cache for next layer

    # next hidden layer down
    layer = 1
    nb_width = width - 1
    z_input = z[layer - 1]
    ds_dw = np.transpose(np.tile(z_input, [1, nb_width]))
    z_output = z[layer][:-1]
    dz_dw = z_output * (1 - z_output) * ds_dw # see q3  
    dL_dz = dz_dz.T @ dL_dz
    dL_dw = dL_dz * dz_dw
    dw[layer] = dL_dw

    # dont need to cache for next layer as we have finished

    return

def propagate(x, y, width, z, w, d_w):
    forward(x, z, w)
    back(y, width, z, w, d_w)

def neural_network_gaussian(D, test, x_width, hl_width, T, r_0, d):
    w = [None] * 4 # weights
    d_w = [None] * 4 # derivative of weights cache for back propagation

    # this will change for the other method
    # intizalize weights with random numbers generated from the standard gaussian distribution 
    # we dont fill in w[0] as there is no weights in the neural network image with w^0_xx
    w[1] = np.random.normal(0, 1, (hl_width - 1, x_width)) 
    w[2] = np.random.normal(0, 1, (hl_width - 1, hl_width))
    w[3] = np.random.normal(0, 1, (1, hl_width))

    # intizalize derivatives of weight to 0
    d_w[1] = np.zeros([hl_width - 1, x_width])
    d_w[2] = np.zeros([hl_width - 1, hl_width])
    d_w[3] = np.zeros([1, hl_width])

    # intizalize z (nodes of the network) values to 1 
    # the last element of z will contain the output
    z = []
    for i in [x_width, hl_width, hl_width, 1]:
        z.append(np.ones([i, 1]))

    # run stochastic gradient descent to learn the neural network
    r = r_0
    for t in range(T):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            propagate(x_i[:-1].reshape([x_width, 1]),x_i[-1].reshape([1,1]), hl_width, z, w, d_w)
            for i in range(1, 4):
                w[i] = w[i] - r * d_w[i]

        r = r_0 / (1 + ((r_0/d) * t))

    # predictions
    # get every output for each sample with final calculated weights
    train_predictions = []
    for x_i in D.values:
        output = forward(x_i[:-1].reshape([5, 1]), z, w)
        train_predictions.append(np.transpose(output))

    # since train_predictions is now an array of rows of the outputs for each value we must concatenate it
    train_predictions = np.concatenate(train_predictions, axis=0).ravel()
    for i in range(0, len(train_predictions)):
        train_predictions[i] = 1 if train_predictions[i] > 0.0 else -1

    test_predictions = []
    for x_i in test.values:
        output = forward(x_i[:-1].reshape([5, 1]), z, w)
        test_predictions.append(np.transpose(output))

    test_predictions = np.concatenate(test_predictions, axis=0).ravel()
    for i in range(0, len(test_predictions)):
        test_predictions[i] = 1 if test_predictions[i] > 0.0 else -1

    return np.mean(train_predictions == D['label']), np.mean(test_predictions == test['label'])

def neural_network_zero(D, test, x_width, hl_width, T, r_0, d):
    w = [None] * 4 # weights
    d_w = [None] * 4 # derivative of weights cache for back propagation

    # this will change for the other method
    # intizalize weights with random numbers generated from the standard gaussian distribution 
    w[1] = np.zeros((hl_width - 1, x_width)) 
    w[2] = np.zeros((hl_width - 1, hl_width))
    w[3] = np.zeros((1, hl_width))

    # intizalize derivatives of weight to 0
    d_w[1] = np.zeros([hl_width - 1, x_width])
    d_w[2] = np.zeros([hl_width - 1, hl_width])
    d_w[3] = np.zeros([1, hl_width])

    # intizalize z (nodes of the network) values to 1 
    # the last element of z will contain the output
    z = []
    for i in [x_width, hl_width, hl_width, 1]:
        z.append(np.ones([i, 1]))

    # run stochastic gradient descent to learn the neural network
    r = r_0
    for t in range(T):
        D_ = D.sample(frac=1, random_state=1)
        for x_i in D_.values:
            propagate(x_i[:-1].reshape([x_width, 1]),x_i[-1].reshape([1,1]), hl_width, z, w, d_w)
            for i in range(1, 4):
                w[i] = w[i] - r * d_w[i]

        r = r_0 / (1 + ((r_0/d) * t))

    # predictions
    # get every output for each sample with final calculated weights
    train_predictions = []
    for x_i in D.values:
        output = forward(x_i[:-1].reshape([5, 1]), z, w)
        train_predictions.append(np.transpose(output))

    # since train_predictions is now an array of rows of the outputs for each value we must concatenate it
    train_predictions = np.concatenate(train_predictions, axis=0).ravel()
    for i in range(0, len(train_predictions)):
        train_predictions[i] = 1 if train_predictions[i] > 0.0 else -1

    test_predictions = []
    for x_i in test.values:
        output = forward(x_i[:-1].reshape([5, 1]), z, w)
        test_predictions.append(np.transpose(output))

    test_predictions = np.concatenate(test_predictions, axis=0).ravel()
    for i in range(0, len(test_predictions)):
        test_predictions[i] = 1 if test_predictions[i] > 0.0 else -1

    return np.mean(train_predictions == D['label']),  np.mean(test_predictions == test['label'])

def run_neural_network():
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

    l0_width = 5 # amount of nodes in layer 0 
    r_0 = 0.1
    d = 0.1
    epoch = 100
    widths = [5, 10, 25, 50, 100]

    print("Neural Network Predictions")
    print("Edge Weights Initialized With Random Numbers From Gaussian Distribution")
    for w in widths:
        print(f"width: {w}") 
        train_e, test_e = neural_network_gaussian(train, test, l0_width, w, epoch, r_0, d)
        print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
        print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
        print()
    print("Edge Weights Initialized With 0")
    for w in widths:
        print(f"width: {w}") 
        train_e, test_e = neural_network_zero(train, test, l0_width, w, epoch, r_0, d)
        print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
        print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
        print()

    return

#endregion
    

#region pytorch neural network

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):
    
    def __init__(self, w, d, T=1000, activation_func=True):
        super(Net, self).__init__()      
        self.T = T          

        # different intilizations depending on activation function
        self.inital_weights = nn.init.xavier_normal_ if not activation_func else nn.init.kaiming_uniform_ 

        def init_weights(model):
            if isinstance(model, nn.Linear):
                self.inital_weights(model.weight)
                torch.nn.init.ones_(model.bias)

        # generate linear layers based off depth and width
        linear_layers = [nn.Linear(5, w), nn.ReLU() if activation_func else nn.Tanh()]
        for _ in range(d-1):
            linear_layers.append(nn.Linear(w, w))
            linear_layers.append(nn.ReLU() if activation_func else nn.Tanh())
        linear_layers.append(nn.Linear(w, 1))

        self.model = nn.Sequential(*linear_layers)

        self.model.apply(init_weights)

    def forward(self, x):    
        input_x = np.float32(x.copy())           
        output = torch.from_numpy(input_x)
        output.requires_grad = True        
        return self.model(output)
            

    def train(self, train_x, train_y):
        train_y = np.float32(train_y.copy())
        train_y = torch.from_numpy(train_y)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters()) # use Adam 
        
        for _ in range(self.T): # epochs
            optimizer.zero_grad()
            output = self.forward(train_x) # forward prop
            loss = criterion(output, train_y)
            loss.backward() # back prop
            optimizer.step()


def neural_network_pytorch(train, test, w, d, function):
    train_y = train.label.values.reshape(-1, 1)
    train_x = train.drop(columns=['label'])
    test_y = test.label.values.reshape(-1, 1)
    test_x = test.drop(columns=['label'])

    model = Net(w=w, d=d, activation_func=function)
    model.train(train_x.values.reshape((-1, train_x.shape[1])), train_y)

    train_predictions = model.forward(train_x).detach().numpy()
    for i in range(0, len(train_predictions)):
        train_predictions[i] = 1 if train_predictions[i] > 0.0 else -1

    test_predictions = model.forward(test_x).detach().numpy()
    for i in range(0, len(test_predictions)):
        test_predictions[i] = 1 if test_predictions[i] > 0.0 else -1

    return np.mean(train_predictions == train_y), np.mean(test_predictions == test_y)

def run_neural_network_pytorch():
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

    activation_functions = [True, False] # true -> ReLU, false -> tahn
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]

    print("Pytorch Neural Network Predictions")
    for function in activation_functions:
        if (function):
            print("Activation Function: ReLU")
        else:
            print("Activation Function: tahn")
        for d in depths:
            print(f"Depth: {d}")
            for w in widths: 
                print(f"Width: {w}")
                train_e, test_e = neural_network_pytorch(train, test, w, d, function)
                print(f"Average Train Prediction Error: {np.around(1-train_e,4)}")
                print(f"Average Test Prediction Error: {np.around(1-test_e,4)}")
                print()

    return
#endregion

def main():
    run_neural_network()
    run_neural_network_pytorch()

main()
