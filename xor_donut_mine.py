# my code-along for the xor_donut (nonlinear classifiers) file.

import numpy as np
import matplotlib.pyplot as plt

# for both XR and donut problem, its binary classification
# so not using softmax

# define the forward action of the network
def forward(X, W1, b1, W2, b2):
    # slightly different then forward & backprop
    # assume tanh() on hidden, softmax on output
    #Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    #activation = Z.dot(W2) + b2
    #Y = 1 / (1+ np.exp(-activation))
    # relu
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z
    return Y, Z

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y) # no argmax

# now for gradient calculations (all "fast" way)
# recall - outer layer 1st, inner layer second

def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()

def derivative_w1(X, Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)

def derivative_b1(Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)

# for cost, we're doing binary cross entropy
def cost(T, Y):
    return np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y))

# XOR
def test_xor():
    # XOR data (see written problem)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0])
    # random weights to start
    W1 = np.random.randn(2,4)
    b1 = np.random.randn(4) # 4 hidden units
    W2 = np.random.randn(4)
    b2 = np.random.randn(1) # only one output node
    # to keep track of likelihoods
    LL = []
    learning_rate = 0.0005
    regularization = 0.0
    last_error_rate = None
    for i in range(100000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        err = np.abs(prediction - Y).mean()
        # error rate prints when changes
        if err != last_error_rate:
            last_error_rate = err
            print('error rate: ', err, ' true: ', Y, ' pred: ', prediction)
        # exit early if log likelihood increases
        if LL and ll < LL[-1]:
            print('early exit')
            break
        LL.append(ll)
        # update weights & biases (gradient descent)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        # print likelihood every 10000 iterations
        if i % 10000 == 0:
            print(ll)
    print('final classification rate:', 1 - np.abs(prediction - Y).mean())
    plt.plot(LL)
    plt.show()

# donut
def test_donut():
    N = 1000
    R_inner = 5  # inner radius
    R_outer = 10 # outer radius
    # distance from origin = radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    # random weights and biases to start, 8 hidden layers
    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    # to keep track of likelihood
    LL = []
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None
    for i in range(160000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)

        # get gradients
        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)

        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)
        if i % 300 == 0:
            print("i:", i, "ll:", ll, "classification rate:", 1 - er)
    plt.plot(LL)
    plt.show()

def test_donut2():
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = [] # keep track of log-likelihoods
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None
    for i in range(3000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)

        # get gradients
        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)

        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)
        if i % 300 == 0:
            print("i:", i, "ll:", ll, "classification rate:", 1 - er)
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    #test_xor()
    test_donut2()



