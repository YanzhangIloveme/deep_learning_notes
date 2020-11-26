# #%% Tensorflow 引入
#
#
# import numpy as np
# import tensorflow as tf
#
# coefficient = np.array([[1.],[-10.],[25.]])
#
# w = tf.Variable(0, dtype=tf.float32)
# x = tf.placeholder(tf.float32, [3,1])
#
# # cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
# # cost = w**2 -10*w +25 # 也可以
#
# cost = x[0][0] * w **2 + x[1][0]*w + x[2][0]
# train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
# init = tf.global_variables_initializer()
# session = tf.Session()
# session.run(init)
# print(session.run(w))
#
# # 输入数据；
# session.run(train, feed_dict={x:coefficient})
# print(session.run(w))
# for i in range(1000):
#     session.run(train,feed_dict={x:coefficient})
# print(session.run(w))

#%% TensorFlow Tutorial¶

# Agenda:
# 1 Initialize variables
# 2 Start your own session
# 3 Train algorithms
# 4 Implement a Neural Network

import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from datasets.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
np.random.seed(1)

#%% 0.1 计算损失函数 $$loss = \mathcal{L}(\hat{y}, y) = (\hat y^{(i)} - y^{(i)})^2 \tag{1}$$ 平方差损失函数

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39
loss = tf.Variable((y_hat-y)**2, name='loss')    # Create a variable for the loss
init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:
    session.run(init)
    print(session.run(loss))                     # Prints the loss

"""
Summary:

Writing and running programs in TensorFlow has the following steps:

1、Create Tensors (variables) that are not yet executed/evaluated.
2、Write operations between those Tensors.
3、Initialize your Tensors.
4、Create a Session.
5、Run the Session. This will run the operations you'd written above.

Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities, 
but did not evaluate its value.
To evaluate it, we had to run init=tf.global_variables_initializer(). 
That initialized the loss variable, and in the last line we were finally able to evaluate the value of loss and print its value.

"""

#%% example
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    print(session.run(c))

#%% 0.2 placeholder

# A placeholder is an object whose value you can specify only later.
# To specify values for a placeholder, you can pass in values by using a "feed dictionary" (feed_dict variable).
# Below, we created a placeholder for x.
x = tf.placeholder(tf.int64, name = 'x')
sess = tf.Session()
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

# When you first defined x you did not have to specify a value for it.
# A placeholder is simply a variable that you will assign data to only later,
# when running the session. We say that you feed data to these placeholders when running the session.

#%% 1.1 - Linear function¶

# Lets start this programming exercise by computing the following equation: $Y = WX + b$, where $W$ and $X$ are random matrices and b is a random vector.
"""
 Compute $WX + b$ where $W, X$, and $b$ are drawn from a random normal distribution. 
 W is of shape (4, 3), X is (3,1) and b is (4,1).
 As an example, here is how you would define a constant X that has shape (3,1):
 X = tf.constant(np.random.randn(3,1), name = "X")
 =============
 You might find the following functions helpful:
    tf.matmul(..., ...) to do a matrix multiplication
    tf.add(..., ...) to do an addition
    np.random.randn(...) to initialize randomly
"""


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)
    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)
    Y = tf.add(tf.matmul(W, X), b)
    with tf.Session() as sess:
        result = sess.run(Y)

    return result

print( "result = " + str(linear_function()))

#%% 1.2 - Computing the sigmoid¶
# Great! You just implemented a linear function. Tensorflow offers a variety of commonly used neural network functions
# like tf.sigmoid and tf.softmax. For this exercise lets compute the sigmoid function of an input.

# You will do this exercise using a placeholder variable x. When running the session,
# you should use the feed dictionary to pass in the input z. In this exercise, you will have to
# (i) create a placeholder x, (ii) define the operations needed to compute the sigmoid using tf.sigmoid, and then (iii) run the session.

# Exercise : Implement the sigmoid function below. You should use the following:

# tf.placeholder(tf.float32, name = "...")
# tf.sigmoid(...)
# sess.run(..., feed_dict = {x: z})


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name="x")
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
    ### END CODE HERE ###

    return result


print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

#%%  1.3 - Computing the Cost

# You can also use a built-in function to compute the cost of your neural network. So instead of needing to write code to
# compute this as a function of cross-entropy error;

# you can do it in one line of code in tensorflow!

# tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)

# Your code should input z, compute the sigmoid (to get a) and then compute the cross entropy cost $J$.
# All this can be done using one call to tf.nn.sigmoid_cross_entropy_with_logits


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    ### START CODE HERE ###
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        result = sess.run(cost, feed_dict={z:logits, y:labels})

    return result

logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = cost(logits, np.array([0, 0, 1, 1]))
print ("cost = " + str(cost))

#%% 1.4 - Using One Hot encodings¶

# Many times in deep learning you will have a y vector with numbers ranging from 0 to C-1, where C is the number of classes.
# If C is for example 4, then you might have the following y vector which you will need to convert
# This is called a "one hot" encoding, because in the converted representation exactly one element of each column is "hot" (meaning set to 1).
# To do this conversion in numpy, you might have to write a few lines of code. In tensorflow, you can use one line of code:

# tf.one_hot(labels, depth, axis)

# Exercise: Implement the function below to take one vector of labels and the total number of classes $C$,
# and return the one hot encoding. Use tf.one_hot() to do this.

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    with tf.Session() as sess:
        result = sess.run(one_hot_matrix)
    return result


labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C=4)
print ("one_hot = " + str(one_hot))

#%% 1.5 - Initialize with zeros and ones¶
# Now you will learn how to initialize a vector of zeros and ones. The function you will be calling is tf.ones().
# To initialize with zeros you could use tf.zeros() instead.
# These functions take in a shape and return an array of dimension shape full of zeros and ones respectively.

def ones(shape):
    ones = tf.ones(shape)
    with tf.Session() as sess:
        result = sess.run(ones)
    return result


print ("ones = " + str(ones([3])))


#%% 2 - Building your first neural network in tensorflow¶

# In this part of the assignment you will build a neural network using tensorflow.

# 2.0 - Problem statement: SIGNS Dataset¶
# One afternoon, with some friends we decided to teach our computers to decipher sign language. We spent a few hours taking pictures in front of a white wall and came up with the following dataset. It's now your job to build an algorithm that would facilitate communications from a speech-impaired person to someone who doesn't understand sign language.
#
# Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
# Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 9
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# As usual you flatten the image dataset, then normalize it by dividing by 255.
# On top of that, you will convert each label to a one-hot vector as shown in Figure 1.
# Run the cell below to do so.


# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

"""
number of training examples = 1080
number of test examples = 120
X_train shape: (12288, 1080)
Y_train shape: (6, 1080)
X_test shape: (12288, 120)
Y_test shape: (6, 120)
"""

# Your goal is to build an algorithm capable of recognizing a sign with high accuracy. To do so, you are going to build
# a tensorflow model that is almost the same as one you have previously built in numpy for cat recognition
# (but now using a softmax output). It is a great occasion to compare your numpy implementation to the tensorflow one.

# The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.
# The SIGMOID output layer has been converted to a SOFTMAX. A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.


#%% 2.1 - Create placeholders¶
# Your first task is to create placeholders for X and Y.
# This will allow you to later pass your training data in when you run your session.

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    ### END CODE HERE ###

    return X, Y

X, Y = create_placeholders(12288, 6)
print("X = " + str(X))
print("Y = " + str(Y))

#%% 2.2 - Initializing the parameters¶

# Your second task is to initialize the parameters in tensorflow.
# Exercise: Implement the function below to initialize the parameters in tensorflow.
# You are going use Xavier Initialization for weights and Zero Initialization for biases.
# The shapes are given below. As an example, to help you, for W1 and b1 you could use:

# W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
# b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25,12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25,1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12,1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6,1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"]))

#%% 2.3 - Forward propagation in tensorflow¶
# You will now implement the forward propagation module in tensorflow.
# The function will take in a dictionary of parameters and it will complete the forward pass.
# The functions you will be using are:

# tf.add(...,...) to do an addition
# tf.matmul(...,...) to do a matrix multiplication
# tf.nn.relu(...) to apply the ReLU activation


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


#%% 2.4 Compute cost¶
# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
# It is important to know that the "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits are expected
# to be of shape (number of examples, num_classes). We have thus transposed Z3 and Y for you.
# Besides, tf.reduce_mean basically does the summation over the examples.

def compute_cost(Z3, Y, alpha):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                       if 'b' not in v.name]) * alpha
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + lossL2

    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y,0.01)
    print("cost = " + str(cost))


#%% 2.5 - Backward propagation & parameter updates¶

# After you compute the cost function. You will create an "optimizer" object.
# You have to call this object along with the cost when running the tf.session.
# When called, it will perform an optimization on the given cost with the chosen method and learning rate.

# For instance, for gradient descent the optimizer would be:
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

# To make the optimization you would do:
# _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
# This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs.

#%% 2.6 - Building the model¶
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,alpha=0.01,
          num_epochs=1600, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    alpha -- L2 regularization with defalut value 0.01

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    ### START CODE HERE ### (1 line)
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    # Initialize parameters
    parameters = initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y,alpha)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)

#%% DIY: 使用固化后的parameters预测

def my_trained_model(X_test, Y_test, parameters):
    (n_x, m) = X_test.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_test.shape[0]
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    X, Y = create_placeholders(n_x, n_y)
    Z3 = forward_propagation(X, parameters)
    A3 = tf.argmax(Z3)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        result = sess.run(A3, feed_dict={X:X_test})


    return result

my_rst = my_trained_model(X_test, Y_test, parameters)
my_rst = pd.DataFrame(my_rst.reshape(-1),columns=['pred'])
my_rst['True'] = np.squeeze(Y_test_orig)


#%% upload my own pic
import scipy
from scipy import ndimage

## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "datasets/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
my_image_prediction = predict(my_image, parameters)

print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))































