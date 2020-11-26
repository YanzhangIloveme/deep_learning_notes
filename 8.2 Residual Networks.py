import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from datasets.resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#%% 1 The problem of very deep neural networks

"""
The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many 
different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, 
using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often
have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. More specifically, during
gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step,
and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values).
"""

#%% 2 - Building a Residual Network

# In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers

# We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn
# an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance.
# (There is also some evidence that the ease of learning an identity function--even more than skip connections helping with vanishing gradients--
# accounts for ResNets' remarkable performance.)

# Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different.
# You are going to implement both of them.

# 2.1 The identity block
# The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$)
# has the same dimension as the output activation (say $a^{[l+2]}$).

# In this exercise, you'll actually implement a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. ' \

"""
steps:

【First component of main path】
The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2a'. 
Use 0 as the seed for the random initialization.
The first BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2a'.
Then apply the ReLU activation function. This has no name and no hyperparameters.

【Second component of main path】
The second CONV2D has $F_2$ filters of shape $(f,f)$ and a stride of (1,1). Its padding is "same" and its name should be conv_name_base + '2b'. Use 0 as the seed for the random initialization.
The second BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2b'.
Then apply the ReLU activation function. This has no name and no hyperparameters.

【Third component of main path】
The third CONV2D has $F_3$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2c'. Use 0 as the seed for the random initialization.
The third BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.

【Final step】
The shortcut and the input are added together.
Then apply the ReLU activation function. This has no name and no hyperparameters.

"""

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X

#
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#     print("out = " + str(out[0][1][1][0]))
#%% 2.2 The convolutional block

# You've implemented the ResNet identity block. Next, the ResNet "convolutional block" is the other type of block.

# You can use this type of block when the input and output dimensions don't match up.
# he difference with the identity block is that there is a CONV2D layer in the shortcut path

"""
setps:

【First component of main path】
The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be 
conv_name_base + '2a'.
The first BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2a'.
Then apply the ReLU activation function. This has no name and no hyperparameters.

【Second component of main path】
The second CONV2D has $F_2$ filters of (f,f) and a stride of (1,1). Its padding is "same" and it's name should be conv_name_base + '2b'.
The second BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2b'.
Then apply the ReLU activation function. This has no name and no hyperparameters.

【Third component of main path】
The third CONV2D has $F_3$ filters of (1,1) and a stride of (1,1). Its padding is "valid" and it's name should be conv_name_base + '2c'.
The third BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.

【Shortcut path】
The CONV2D has $F_3$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '1'.
The BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '1'.

"""

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X
    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s, s),padding='valid',name=conv_name_base+'2a',kernel_initializer=
               glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+"2b",kernel_initializer=
               glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1),strides=(s,s),padding = 'valid',name=conv_name_base+"1",kernel_initializer=
                        glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

#%% 3 - Building your first ResNet model (50 layers)¶

# The details of this ResNet-50 model are:

"""
Zero-padding pads the input with a pad of (3,3)

【stage 1】
The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
BatchNorm is applied to the channels axis of the input.
MaxPooling uses a (3,3) window and a (2,2) stride.
【Stage 2】
The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
【Stage 3】
The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
【Stage 4】
The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
【Stage 5】
The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
The flatten doesn't have any hyperparameters or name.
The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).

"""

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7,7),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
#%% 训练模型

model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

#%% 调用已经训练好的模型
model = load_model("models/ResNet50.h5")
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

#%% test my own
img_path = 'datasets/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = plt.imread(img_path)
plt.imshow(my_image)
plt.show()
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))

#%%
model.summary()
plot_model(model, to_file='model.png')
