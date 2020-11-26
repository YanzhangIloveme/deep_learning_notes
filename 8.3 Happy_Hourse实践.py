import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from datasets.kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


#%%  case介绍：
"""
本次我们将
1 学习到一个高级的神经网络的框架，能够运行在包括TensorFlow和CNTK的几个较低级别的框架之上的框架。
2 看看如何在几个小时内建立一个深入的学习算法。

任务描述：下一次放假的时候，你决定和你的五个朋友一起度过一个星期。这是一个非常好的房子，在附近有很多事情要做，但最重要的好处是每个人在家里都会感到快乐，
所以任何想进入房子的人都必须证明他们目前的幸福状态。作为一个深度学习的专家，为了确保“快乐才开门”规则得到严格的应用，你将建立一个算法，
它使用来自前门摄像头的图片来检查这个人是否快乐，只有在人高兴的时候，门才会打开。
你收集了你的朋友和你自己的照片，被前门的摄像头拍了下来。数据集已经标记好了。。

"""
#%% 数据处理

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# plt.imshow(X_train_orig[0])
# plt.show()
# normalize

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# Keras 建模大纲：


"""
def model(input_shape):
	"""
	# 模型大纲
	"""
    #定义一个tensor的placeholder，维度为input_shape
    X_input = Input(input_shape)
    
    #使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3,3))(X_input)
    
    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    #最大值池化层
    X = MaxPooling2D((2,2),name="max_pool")(X)
    
    #降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    #创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model
"""

#%% Model

def HappyModel(input_shape):
    """
    实现简单的笑脸检测算法
    :param input_shape: 输入的数据维度
    :return: 创建keras的模型
    """

    X_input = Input(input_shape)
    # 使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3,3))(X_input)
    # 对x使用conv-bn-relu块
    X = Conv2D(32, (7,7), strides=(1,1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='BN0')(X)
    X = Activation('relu')(X)

    # 最大化池化
    X = MaxPooling2D((2,2), name='max_pool')(X)

    # 降维
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name = 'HappyModel')

    return model

"""
创建一个模型实体。
编译模型，可以使用这个语句：model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])。
训练模型：model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)。
评估模型：model.evaluate(x = ..., y = ...)。
"""

happy_model = HappyModel(X_train.shape[1:])
happy_model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
# 训练模型
happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print ("误差值 = " + str(preds[0]))
print ("准确度 = " + str(preds[1]))

# 1.5 - 其他一些有用的功能
# model.summary()：打印出你的每一层的大小细节
# plot_model() : 绘制出布局图