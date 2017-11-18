#  code based on https://iamtrask.github.io/2015/07/12/basic-python-network/

# this is to solve exercise in http://neuralnetworksanddeeplearning.com/chap1.html
# given 10 input nodes 0..9 with output that converts the numbers to binary values of 4 bits.
# derive weight for each of the 4 output nodes
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1] ])
    
# output dataset            
all_y=np.array( [[0,0,0,0],
                [0,0,0,1],
                [0,0,1,0],
                [0,0,1,1],
                [0,1,0,0],
                [0,1,0,1],
                [0,1,1,0],
                [0,1,1,1],
                [1,0,0,0],
                [1,0,0,1]])


# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((10,1)) - 1

for o in range(0,all_y.shape[1]):
    y = np.array([[c[o] for c in all_y]]).T
    print("training for output node {0}".format(o))
    for iter in range(10000):

        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1,True)

    # update weights
        syn0 += np.dot(l0.T,l1_delta)

    print( "Output After Training:")
    print(l1)

