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
def nettrain(train_xs, train_ys, iterations):
    #return weights#
    w = [[[],[],[]],[[],[]]] #a two layer net will do I believe#
    for i in train_xs[0]:
        w[0][0].append(np.float64(0.01))
        w[0][1].append(np.float64(-0.01))
        w[0][2].append(np.float64(0.01))
        #first connection has three weights matrix#
        w[1][0].append(np.float64(0.01))
        w[1][1].append(np.float64(0.01))
    #training part#
    accuracy = []
    for i in range(0,iterations):
        for j in range(0,len(train_xs)):
            t = []#this is specific for this dataset#
            if(train_ys[j] == 1):
                t = [1,0]
            else:
                t = [0,1]
            w = backprop(w,train_xs[j],t)
        acc = test_accuracy(w,train_xs,train_ys)
        accuracy.append(acc)
        if(acc >= 1.0000000):#converged
            return w,accuracy
    return w,accuracy

def backprop(w,x,t):
    out,y = propagation(w,x)
    error = []
    for i in range(0,len(y)):
        error.append(np.float64(0))
        error[i] = t[i] - y[i]
    #note that dY = error#
    #Perforn back prop!#
    
    
    return w

#using Relu but this can be adapted into Sigmoid or Tanh since outputs of each layer are recorded.
def propagation(w,x):
    out = []
    for i in range(0,len(w)):
        out.append([])
        for j in range(0,len(w[i])):
            out[i].append(np.float64(0))
    for i in range(0,len(w)):
        lastlayer = []
        if (i == 0):
            lastlayer = x
        else:
            lastlayer = out[i-1]
        for j in range(0,len(w[i])):
            out[i][j] = np.dot(lastlayer, w[i][j])
            if(out[i][j] <= 0):
                out[i][j] = 0
    return out,out[(len(out)-1)]

def classify(w,x):
    out,y = propagation(w,x)
    if y[0] > y[1]:
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
    
    args = parser.parse_args()
    
    train_file = "C:/Users/yyu56/Desktop/XorNet/data/xorSmoke.dat"
    iterations = 10
    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(train_file)
    
    weights,accuracy = nettrain(train_xs, train_ys, iterations)
    accuracy = test_accuracy(weights, train_xs, train_ys)
    print('Train accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))
    
if __name__ == '__main__':
    main()