# my code-along for the backpropagation file.

import numpy as np
import matplotlib.pyplot as plt

# copied from forward prop:

# define the forward action of the network
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

# define a function for classification rate
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:   # if predicted = true, increase n_correct
            n_correct += 1
    return float(n_correct) / n_total

# new functions

# derivative wrt W2
def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    # slow way
    #ret1 = np.zeros((M,K))
    #for n in range(N):
    #    for m in range(M):
    #        for k in range(K):
    #            ret1[m,k] += (T[n,k] - Y[n,k]) * Z[n,m]

    # fast way
    #ret2 = np.zeros((M,K))
    #for n in range(N):
    #    for k in range(K):
    #        ret2[:,k] += (T[n,k] - Y[n,k]) * Z[n,:]

    # faster (only looping through N
    #ret3 = np.zeros((M,K))
    #for n in range(N):
    #    ret3 += np.outer(Z[n], T[n] - Y[n])

    # simply even further (no looping!)
    #ret4 = Z.T.dot(T-Y)

    return Z.T.dot(T-Y)

# derivative wrt b2
def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

# derivative wrt W1
def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    # slow way
    #ret1 = np.zeros((D,M))
    #for n in range(N):
    #    for k in range(K):
    #        for m in range(M):
    #            for d in range(D):
    #                ret1[d,m] += (T[n,k] - Y[n,k]) * W2[m,k] * \
    #                             Z[n,m]*(1 - Z[n,m]) * X[n,d]

    # fast way
    dZ = (T-Y).dot(W2.T) * Z * (1-Z)
    return X.T.dot(dZ)

    return ret1

# derivative wrt b2
def derivative_b1(T, Y, W2, Z):
    # fast way!
    return((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)

# cost function
def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

# main function
def main():
    # create the data
    Nclass = 500  # number of classes
    D = 2  # dimension of input arrays
    M = 3  # number of hidden layers
    K = 3  # number of classes

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])  # Gaussian cloud, centered at (0,-2)
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])  # Gaussian cloud, centered ay (2,2)
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])  # Gaussian cloud, centered at (-2,2)
    X = np.vstack([X1, X2, X3])  # combine clouds into a matrix
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)  # labels
    N = len(Y)

    # indicator matrix - one hot encoding for targets
    T = np.zeros((N,K))
    for i in range(N):
        T[i, Y[i]] = 1

    # visualize
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

    # randomly initialize the weights & biases
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    # now, backprop
    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        # every 100 epochs, calculate the cost
        if epoch % 100 == 0:
            c = cost(T, output) # cost
            P = np.argmax(output, axis=1) # prediction
            r = classification_rate(Y, P) # classification rate
            print('cost: ', c, 'classification rate: ', r)
            costs.append(c)

        # calculation of weights (2 first, going backward)
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    # now, plot cost
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()