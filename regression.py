# logistic regression 

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# generate and plot the data
# 500 evenly space points bw -2 and 2 on 2D grid
N = 500
X = np.random.random((N,2))*4 - 2
Y = X[:,0] * X[:,1] # saddle shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# make a NN to fit to this dataset & train it
D = 2    # dimension of the input
M = 100  # number of hidden units / layers

# layer 1
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# layer 2
V = np.random.randn(M) / np.sqrt(M)
c = 0

# function to calculate the output
def forward(X):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # relu
    Yhat = Z.dot(V) + c
    return Z, Yhat

# functions to calculate derivatives for gradient descent
def derivative_V(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)

def derivative_c(Y, Yhat):
    return (Y - Yhat).sum()

def derivative_W(X, Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # relu
    return X.T.dot(dZ)

def derivative_b(Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # relu
    return dZ.sum(axis=0)

def update(X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-4):
    # gradients
    gV = derivative_V(Z, Y, Yhat)
    gc = derivative_c(Y, Yhat)
    gW = derivative_W(X, Z, Y, Yhat, V)
    gb = derivative_b(Z, Y, Yhat, V)
    # gradient descent
    V += learning_rate*gV
    c += learning_rate*gc
    W += learning_rate*gW
    b += learning_rate*gb

    return W, b, V, c

# for plotting costs (MSE)
def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()

# next, a training loop to preform updates and keeps track of costs
costs = []
for i in range(200):
    Z, Yhat = forward(X)
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    cost = get_cost(Y, Yhat)
    costs.append(cost)
    if i % 25 == 0:
        print('cost = ', cost)

# plot costs
plt.plot(costs)
plt.show()

# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# surface
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()

# plot magnitude of the residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)
plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()





