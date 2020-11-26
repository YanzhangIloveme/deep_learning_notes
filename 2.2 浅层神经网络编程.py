#%% packages

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from datasets.testCases import *
from datasets.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

#%% Simple Logistic Regression
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")

#%% NN
# 1 设置size
def layer_sizes(X, Y):
    # n_x: size of input layer
    # n_h: size of hidden layer
    # n_y: size of output layer
    n_x = X.shape[0]   # 2
    n_h = 4            # 4
    n_y = Y.shape[0]   # 1
    return n_x, n_h, n_y

# X_assess, Y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)

#%% 2 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))
    params = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return params

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#%% 3 向前传播
def forward_propagation(X, parameters):
    # X -- input data of size (n_x, m)
    # parameters -- python dictionary containing your parameters (output of initialization function)
    # Returns:
    # A2 -- The sigmoid output of the second activation
    # cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    W1 = parameters['W1']  # (n_h, n_x)
    b1 = parameters['b1']  # (n_h, 1)
    W2 = parameters['W2']  # (n_y, n_h)
    b2 = parameters['b2']  # (n_y, 1)
    Z1 = np.dot(W1, X) + b1  # (n_h, n_x) * (n_x, m) + (n_h, 1) = (n_h, m)
    A1 = np.tanh(Z1)  # (n_h, m)
    Z2 = np.dot(W2, A1) + b2 #  (n_y, n_h) *  (n_h, m) = (n_y, m)
    A2 = sigmoid(Z2) # (n_y, m)
    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2, cache

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# # Note: we use the mean here just to make sure that your output matches ours.
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

#%% 4 反向传播part1：损失函数
def compute_cost(A2, Y, parameters):
    # Computes the cross-entropy cost
    m = Y.shape[1]
    logprobs = - (np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2)))
    cost = np.sum(logprobs)/m
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect. turns [[17]] into 17
    return cost

A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

#%% 4 反向传播part2：传播
def backward_propagation(parameters, cache, X, Y):
    # return：grads -- python dictionary containing your gradients with respect to different parameters
    m = X.shape[1]
    A1 = cache['A1'] # (n_h,m)
    A2 = cache['A2'] # (n_y, m)
    W1 = parameters['W1'] # (n_h, n_x)
    W2 = parameters['W2'] # (n_y, n_h)
    dZ2 = A2-Y  # (n_y, m)
    dW2 = (1 / m) * np.dot(dZ2, A1.T) # (n_y, n_h)
    db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)  # (n_y, 1)
    dZ1 = np.multiply(np.dot(W2.T, dZ2) , (1-np.power(A1,2)))# (n_h,m) = [(n_h, n_y) * (n_y, m)] * (n_h,m)
    dW1 = (1 / m) *np.dot(dZ1, X.T) # (n_h, n_x) = (n_h,m) * (m, n_x)
    db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)     # (n_h, 1) = sum((n_h,m),axis=1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

# arameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

#%% 5 GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#%% 6  GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations=10000, learning_rate = 1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Loop (gradient descent)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads,learning_rate=learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#%% 7 预测
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    ### END CODE HERE ###

    return predictions

# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))

#%% 8 在数据集上运行我们的模型

parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

#%% 9 我们来探究不同隐藏层对结果对影响

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()
#%% 10 model on different datasets

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y % 2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()

parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
