import time
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/StyleNet/')
import nst_utils


#%% 1 问题描述
"""
神经风格转换（Neural Style Transfer，NST）是深学习中最有趣的技术之一。如下图所示，它合并两个图像，即“内容”图像（C CContent）和“风格”图像（S SStyle），
以创建“生成的”图像（G GGenerated）。生成的图像G将图像C的“内容”与图像S的“风格”相结合。
在这个例子中，你将生成一个巴黎卢浮宫博物馆（内容图像C）与一个领袖印象派运动克劳德·莫奈的画（风格图像S）混合起来的绘画。
"""
#%% 2 迁移学习
# 神经风格转换（NST）使用先前训练好了的卷积网络，并在此基础之上进行构建。使用在不同任务上训练的网络并将其应用于新任务的想法称为迁移学习。
#  根据原始的NST论文(https://arxiv.org/abs/1508.06576 )，我们将使用VGG网络，具体地说，我们将使用VGG-19，这是VGG网络的19层版本。这个模型已经在非常大的ImageNet数据库上进行了训练，因此学会了识别各种低级特征(浅层)和高级特征(深层)。
#  运行以下代码从VGG模型加载参数。这可能需要几秒钟的时间。

model = nst_utils.load_vgg_model("datasets/StyleNet/pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

# 该模型存储在一个python字典中，其中每个变量名都是键，
# 相应的值是一个包含该变量值的张量,要通过此网络运行图像，只需将图像提供给模型。 在TensorFlow中，你可以使用tf.assign函数来做到这一点:
# model["input"].assign(image)

#%% 3 - 神经风格转换
"""
我们可以使用下面3个步骤来构建神经风格转换（Neural Style Transfer，NST）算法：
构建内容损失函数J c o n t e n t ( C , G ) J_{content}(C,G)
构建风格损失函数J s t y l e ( S , G ) J_{style}(S,G)
把它放在一起得到J ( G ) = α J c o n t e n t ( C , G ) + β J s t y l e ( S , G ) J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)J
"""

#%% 3.1 - 计算内容损失
content_image = plt.imread("datasets/StyleNet/images/louvre.jpg")
imshow(content_image)
plt.show()
# 内容图片©显示了卢浮宫的金字塔被旧的巴黎建筑包围，图片上还有阳光灿烂的天空和一些云彩。

#%% 3.2 - 如何确保生成的图像G与图像C的内容匹配?
# 正如我们在视频中看到的，浅层的一个卷积网络往往检测到较低层次的特征，如边缘和简单的纹理，更深层往往检测更高层次的特征，如更复杂的纹理以及对象分类等。
# 我们希望“生成的”图像G具有与输入图像C相似的内容。假设我们选择了一些层的激活来表示图像的内容，在实践中，如果你在网络中间选择一个层——既不太浅也不太深，
# 你会得到最好的的视觉结果。（当你完成了这个练习后，你可以用不同的图层进行实验，看看结果是如何变化的。）

"""
假设你选择了一个特殊的隐藏层，现在，将图像C作为已经训练好的VGG网络的输入，然后进行前向传播
让a[c]成为你选择的层中的隐藏层激活,激活值为[nh, nw, nc]张量
然后用图像G重复这个过程：将G设置为输入数据，并进行前向传播，让a[G]成为相应的隐层激活

现在我们要使用tensorflow来实现内容代价函数，它由以下3步构成：
1 从a_G中获取维度信息：从张量X中获取维度信息，可以使用：X.get_shape().as_list()
2 将a_C与a_G如上图一样降维
3 计算内容代价
"""

def compute_content_cost(a_C, a_G):
    """
    计算内容代价的函数

    参数：
        a_C -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像C的内容的激活值。
        a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的内容的激活值。

    返回：
        J_content -- 实数，用上面的公式1计算的值。

    """

    # 获取a_G的维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # 对a_C与a_G从3维降到2维
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
    # 计算内容代价
    # J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

tf.random.set_seed(1)
a_C = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
J_content = compute_content_cost(a_C, a_G)
print("J_content = " + str(J_content.numpy()))

#%% 3.3 - 计算风格损失
style_image = plt.imread("datasets/StyleNet/images/monet_800600.jpg")

imshow(style_image)
plt.show()

"""
The style matrix is also called a "Gram matrix." In linear algebra, the Gram matrix G of a set of vectors （v1,v2..vn)
is the matrix of dot products, whose entries are Gij=vit*vj = np.dot(vi, vj).
In other words, Gij compares how similar vi is to vj: If they are highly similar, you would expect them to have a large dot product,
and thus for Gi to be large.



"""
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, tf.transpose(A))
    ### END CODE HERE ###

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S))  # 矩阵转置的时候一定要注意方向
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * tf.square(tf.to_float(n_H * n_W * n_C)))

    ### END CODE HERE ###

    return J_style_layer
#%% 3.2.3 Style Weights

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style
#%% 3.3 - Defining the total cost to optimize¶
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###

    return J


np.random.seed(3)
J_content = np.random.randn()
J_style = np.random.randn()
J = total_cost(J_content, J_style)

#%% 4 - Solving the optimization problem
"""
Finally, let's put everything together to implement Neural Style Transfer!

Here's what the program will have to do:

Create an Interactive Session
Load the content image
Load the style image
Randomly initialize the image to be generated
Load the VGG16 model
Build the TensorFlow graph:
Run the content image through the VGG16 model and compute the content cost
Run the style image through the VGG16 model and compute the style cost
Compute the total cost
Define the optimizer and the learning rate
Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step."""