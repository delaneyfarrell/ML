# builds a feed forward NN without using libraries

import numpy as np
import matplotlib.pyplot as plt

# data (X and Y)
Nclass = 500    # number of classes
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])   # Gaussian cloud, centered at (0,-2)
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])    # Gaussian cloud, centered ay (2,2)
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])   # Gaussian cloud, centered at (-2,2)
X = np.vstack([X1,X2,X3])   # combine clouds into a matrix
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)    # labels

# visualize
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# dimensions
D = 2   # dimension of input arrays
M = 3   # number of hidden layers
K = 3   # number of classes

# initialize the weights & biases
W1 = np.random.randn(D, M); b1 = np.random.randn(M)
W2 = np.random.randn(M, K); b2 = np.random.randn(K)

# define the forward action of the network
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

# define a function for classification rate
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:   # if predicted = true, increase n_correct
            n_correct += 1
    return float(n_correct) / n_total

# now call forward with randomly defined variables to get classification rate
P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

# assert we have chosen the right axis to perform argmax
assert(len(P) == len(Y))

print("Classification rate for randomly chosen weights: ", classification_rate(Y, P))
