import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# problem: 3 Gaussian clouds i.e. classes
# solve with TF

# create random training data again
Nclass = 500
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

N = len(Y)
# turn Y into an indicator matrix for training
T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1


# TF doesn't use regular numpy arrays, instead uses "tensorflow variables"
# so the function below wraps input parameters as TF variable
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1))


# takes in input with parameters, define how network computes the output
def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)  # TF does not have dot like np, instead matmul
    return tf.matmul(Z, W2) + b2


# create TF placeholders to hold the data
tf.compat.v1.disable_eager_execution()
tfX = tf.compat.v1.placeholder(tf.float32, [None, D])
tfY = tf.compat.v1.placeholder(tf.float32, [None, K])

# create weights
W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

# forward function gets output of the NN
logits = forward(tfX, W1, b1, W2, b2)

# cost / loss function
cost = tf.reduce_mean(
    tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=tfY,
        logits=logits
    )
)

# operations to train and predict
train_op = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(logits, 1)

# now, actual computation - this has to be done in TF1 (doesnt exist in TF2)
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# main training loop
for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})   # runs one step of backprop
    pred = sess.run(predict_op, feed_dict={tfX: X})  # predict returns a np array
    if i % 100 == 0:
        print('Accuracy', np.mean(Y == pred))
