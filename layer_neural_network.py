import numpy as np


def nonlin(X, deriv=False):
    # Activation function: sigmoid
    if(deriv):
        return X * (1 - X)

    return 1 / (1 + np.exp(-X))


# input
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# output
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation deterministic
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3,1)) - 1

for i in range(10000):

    # foward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    #multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    #update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Ouput after training:")
print(l1)