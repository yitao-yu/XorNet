#!/usr/bin/python3

# AUTHOR:  Yitao Yu
# NetID:   yyu56

#making use of codes that we wrote in HW1

import numpy as np
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys

# Do learning.
def nettrain(train_xs, train_ys, iterations,learning_rate):
    #return weights#
    w = [[[],[]],[[],[]]] #a two layer net will do I believe#
    #This part can be adapted to different topology#
    import random
    for i in train_xs[0]:
        w[0][0].append(np.float64(-1+2*random.random()))
        w[0][1].append(np.float64(-1+2*random.random()))
        
        #first connection has three weights matrix#
        w[1][0].append(np.float64(-1+2*random.random()))
        w[1][1].append(np.float64(-1+2*random.random()))
    #training part#
    accuracy = []
    
    for i in range(0,iterations):
        for j in range(0,len(train_xs)):
            t = []#this is specific for class labels of this dataset#
            if(train_ys[j] == 1):
                t = [1.0,0.0]
            else:
                t = [0.0,1.0]
            w = backprop(w.copy(),train_xs[j],t,learning_rate)
        acc = test_accuracy(w,train_xs,train_ys)
        accuracy.append(acc)
        if(acc >= 1.0000000):#converged
            return w,accuracy
    return w,accuracy

def backprop(w,x,t,lr):
    out,y = propagation(w,x)
    print(y)
    error = []
    #deep copy#
    for i in range(0,len(out)):
        error.append([])
        for j in range(0,len(y)):
            error[i].append(out[i][j])
    for i in range(0,len(error)):
        for j in range(0,len(error[i])):
            error[i][j] = np.float64(0.0)
    
    for i in range(0,len(y)):
        error[len(error)-1][i] = y[i] - t[i]#Loss#
    #Blame each Neuron in Hidden Layer, Using Chain Rule#
    for i in range(len(error)-1, 0,-1):#index for layer
            for j in range(0,len(error[i])):#index for neuron in this layer
                blame = error[i][j] * d_activation(out[i][j])
                for k in range(0,len(error[i-1])):#index for input neuron from last layer
                    d_a = out[i-1][k]
                    error[i-1][k] += blame * d_a
    #Adjust the weight of each connection using gradient descent#
    #for each blame in the error; blame them to each connection to the last layer#
    #basically what should happens: d_activtion/d_w = x#
    #d_blame/d_w = d_blame/d_activation * d_activation/d_w = error*x#
    for i in range(0,len(w)):
        for j in range(0,len(w[i])):
            e = error[i][j]
            for k in range(0,len(w[i][j])):
                blame_w = 0
                if (i==0):
                    blame_w = e * x[k]
                else:
                    blame_w = e * out[i-1][k]
                w[i][j][k] += blame_w * lr
    return w

def d_activation(y):#derivative for Sigmoid#
    return y*(1-y)

#using Sigmoid but this can be adapted into Sigmoid or Tanh since outputs of each layer are recorded.
def propagation(w,x):
    out = []
    for i in range(0,len(w)):
        out.append([])
        for j in range(0,len(w[i])):
            out[i].append(np.float64(0))
    for i in range(0,len(w)):
        lastlayer = []
        if (i == 0):
            lastlayer = x.copy()
        else:
            lastlayer = out[i-1].copy()
        for j in range(0,len(w[i])):
            a = np.dot(lastlayer, w[i][j])
            out[i][j] = 1/(1+np.exp(-a))
        if(i<(len(out)-1)):
            out[i].append(1.0)
    return out.copy(),out[(len(out)-1)].copy()

def classify(w,x):
    out,y = propagation(w,x)
    if y[0] >= y[1]:
        return 1; #the first output is score for class 1
    else:
        return -1;

# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    w = weights
    count = [0,0]
    for i in range(0, len(test_xs)):
        if classify(w,test_xs[i])*test_ys[i] > 0:
            count[0] += 1
        count[1] += 1
    return np.float64(count[0])/np.float64(count[1])
    
def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Training data file.')
    
    args = parser.parse_args()
    
    train_file = "C:/Users/yyu56/Desktop/XorNet-master/data/xorSmoke.dat"
    iterations = 1000
    lr = 0.03056
    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(train_file)
    
    weights,accuracy = nettrain(train_xs, train_ys, iterations,lr)
    accuracy = test_accuracy(weights, train_xs, train_ys)
    print('Train accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))
    
if __name__ == '__main__':
    main()
