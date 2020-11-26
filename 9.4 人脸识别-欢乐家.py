# 引入
import tensorflow as tf
from tensorflow import keras
from datasets.FaceNet import fr_utils
tf.keras.backend.set_image_data_format('channels_first')

"""
人脸识别系统通常被分为两大类：

 人脸验证：“这是不是本人呢？”，比如说，在某些机场你能够让系统扫描您的面部并验证您是否为本人从而使得您免人工检票通过海关，又或者某些手机能够使用人脸解锁功能。这些都是1：1匹配问题。
 人脸识别：“这个人是谁？”，比如说，在视频中的百度员工进入办公室时的脸部识别视频的介绍，无需使用另外的ID卡。这个是1：K的匹配问题

 FaceNet可以将人脸图像编码为一个128位数字的向量从而进行学习，通过比较两个这样的向量，那么我们就可以确定这两张图片是否是属于同一个人。

在本节中，你将学到：

 实现三元组损失函数。
 使用一个已经训练好了的模型来将人脸图像映射到一个128位数字的的向量。
 使用这些编码来执行人脸验证和人脸识别。

在此次练习中，我们使用一个训练好了的模型，该模型使用了“通道优先”的约定来代表卷积网络的激活，而不是在视频中和以前的编程作业中使用的“通道最后”的约定。换句话说，数据的维度是
(m, nc, nh, nw) 而不是 (m, nh, nw, nc)
"""


# 0 - 简单的人脸验证
"""
在人脸验证中，你需要给出两张照片并想知道是否是同一个人，最简单的方法是逐像素地比较这两幅图像，如果图片之间的误差小于选择的阈值，那么则可能是同一个人。

当然，如果你真的这么做的话效果一定会很差，因为像素值的变化在很大程度上是由于光照、人脸的朝向、甚至头部的位置的微小变化等等。
接下来与使用原始图像不同的是我们可以让系统学习构建一个编码$f(img)$，对该编码的元素进行比较，可以更准确地判断两幅图像是否属于同一个人。

"""

# 1 - 图像编码为128位的向量将人脸
#%% 1.1 - 使用卷积网络来进行编码
"""
FaceNet模型需要大量的数据和长时间的训练，因为，遵循在应用深度学习设置中常见的实践，我们要加载其他人已经训练过的权值。
在网络的架构上我们遵循Szegedy et al.等人的初始模型。这里我们提供了初始模型的实现方法，你可以打开inception_blocks.py文件来查看是如何实现的。

该网络使用了96 × 96 的RGB图像作为输入数据，图像数量为m，输入的数据维度为
(m, nc, nh, nw) = ( m, 3, 96, 96 ) 
输出为(m, 128) (m,128)(m,128)的已经编码的m个128位的向量。


通过使用128神经元全连接层作为最后一层，该模型确保输出是大小为128的编码向量，然后使用比较两个人脸图像的编码如下：
因此，如果满足下面两个条件的话，编码是一个比较好的方法：
    同一个人的两个图像的编码非常相似。
    两个不同人物的图像的编码非常不同。
    三元组损失函数将上面的形式实现，它会试图将同一个人的两个图像（对于给定的图和正例）的编码“拉近”，同时将两个不同的人的图像（对于给定的图和负例）进一步“分离”。


"""

#%% 1.2 - 三元组损失函数
#  一些会用到的函数：tf.reduce_sum()， tf.square()， tf.subtract()， tf.add(), tf.maximum()\

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    根据公式（4）实现三元组损失函数

    参数：
        y_true -- true标签，当你在Keras里定义了一个损失函数的时候需要它，但是这里不需要。
        y_pred -- 列表类型，包含了如下参数：
            anchor -- 给定的“anchor”图像的编码，维度为(None,128)
            positive -- “positive”图像的编码，维度为(None,128)
            negative -- “negative”图像的编码，维度为(None,128)
        alpha -- 超参数，阈值

    返回：
        loss -- 实数，损失的值
    """
    #获取anchor, positive, negative的图像编码
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    #第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    #第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    #第三步：减去之前的两个距离，然后加上alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist),alpha)
    #通过取带零的最大值和对训练样本的求和来计算整个公式
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss

# 测试
y_true = (None, None, None)
y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
          tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
          tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
loss = triplet_loss(y_true, y_pred)
print(loss)

#%% 2 - 加载训练好了的模型
#  FaceNet是通过最小化三元组损失来训练的，但是由于训练需要大量的数据和时间，所以我们不会从头训练，相反，我们会加载一个已经训练好了的模型，运行下列代码来加载模型，可能会需要几分钟的时间。

from datasets.FaceNet.inception_blocks_v2 import faceRecoModel

FRmodel = faceRecoModel(input_shape=(3,96,96))
fr_utils.load_weights_from_FaceNet(FRmodel)

#%% 3 - 模型的应用
# 之前我们对“欢乐家”添加了笑脸识别，现在我们要构建一个面部验证系统，以便只允许来自指定列表的人员进入。为了通过门禁，每个人都必须在门口刷身份证以表明自己的身份，然后人脸识别系统将检查他们到底是谁。
"""
我们构建一个数据库，里面包含了允许进入的人员的编码向量，我们使用fr_uitls.img_to_encoding(image_path, model)函数来生成编码，它会根据图像来进行模型的前向传播。
我们这里的数据库使用的是一个字典来表示，这个字典将每个人的名字映射到他们面部的128维编码上。

"""
database = {}
database["danielle"] = fr_utils.img_to_encoding("datasets/FaceNet/images/danielle.png", FRmodel)
database["younes"] = fr_utils.img_to_encoding("datasets/FaceNet/images/younes.jpg", FRmodel)
database["tian"] = fr_utils.img_to_encoding("datasets/FaceNet/images/tian.jpg", FRmodel)
database["andrew"] = fr_utils.img_to_encoding("datasets/FaceNet/images/andrew.jpg", FRmodel)
database["kian"] = fr_utils.img_to_encoding("datasets/FaceNet/images/kian.jpg", FRmodel)
database["dan"] = fr_utils.img_to_encoding("datasets/FaceNet/images/dan.jpg", FRmodel)
database["sebastiano"] = fr_utils.img_to_encoding("datasets/FaceNet/images/sebastiano.jpg", FRmodel)
database["bertrand"] = fr_utils.img_to_encoding("datasets/FaceNet/images/bertrand.jpg", FRmodel)
database["kevin"] = fr_utils.img_to_encoding("datasets/FaceNet/images/kevin.jpg", FRmodel)
database["felix"] = fr_utils.img_to_encoding("datasets/FaceNet/images/felix.jpg", FRmodel)
database["benoit"] = fr_utils.img_to_encoding("datasets/FaceNet/images/benoit.jpg", FRmodel)
database["arnaud"] = fr_utils.img_to_encoding("datasets/FaceNet/images/arnaud.jpg", FRmodel)
#%%
"""
现在，当有人出现在你的门前刷他们的身份证的时候，你可以在数据库中查找他们的编码，用它来检查站在门前的人是否与身份证上的名字匹配。

现在我们要实现 verify() 函数来验证摄像头的照片(image_path)是否与身份证上的名称匹配，这个部分可由以下步骤构成：
根据image_path来计算编码。
计算与存储在数据库中的身份图像的编码的差距。
如果差距小于0.7，那么就打开门，否则就不开门。
"""


def verify(image_path, identity, database, model):
    """
    对“identity”与“image_path”的编码进行验证。

    参数：
        image_path -- 摄像头的图片。
        identity -- 字符类型，想要验证的人的名字。
        database -- 字典类型，包含了成员的名字信息与对应的编码。
        model -- 在Keras的模型的实例。

    返回：
        dist -- 摄像头的图片与数据库中的图片的编码的差距。
        is_open_door -- boolean,是否该开门。
    """
    # 第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    # 第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - database[identity])

    # 第三步：判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open

#%%
# 面部验证系统基本运行良好，但是自从Kian的身份证被偷后，那天晚上他回到房子那里就不能进去了!
# 为了减少这种恶作剧，你想把你的面部验证系统升级成面部识别系统。这样就不用再带身份证了，一个被授权的人只要走到房子前面，前门就会自动为他们打开!

#  我们将实现一个人脸识别系统，该系统将图像作为输入，并确定它是否是授权人员之一(如果是，是谁),与之前的人脸验证系统不同，我们不再将一个人的名字作为输入的一部分。
#  现在我们要实现who_is_it()函数，实现它需要有以下步骤：
"""
根据image_path计算图像的编码。
从数据库中找出与目标编码具有最小差距的编码。
--初始化min_dist变量为足够大的数字（100），它将找到与输入的编码最接近的编码。
--遍历数据库中的名字与编码，可以使用for (name, db_enc) in database.items()语句。
----计算目标编码与当前数据库编码之间的L2差距。
----如果差距小于min_dist，那么就更新名字与编码到identity与min_dist中。

"""


def who_is_it(image_path, database, model):
    """
    根据指定的图片来进行人脸识别

    参数：
        images_path -- 图像地址
        database -- 包含了名字与编码的字典
        model -- 在Keras中的模型的实例。

    返回：
        min_dist -- 在数据库中与指定图像最相近的编码。
        identity -- 字符串类型，与min_dist编码相对应的名字。
    """
    # 步骤1：计算指定图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    # 步骤2 ：找到最相近的编码
    ## 初始化min_dist变量为足够大的数字，这里设置为100
    min_dist = 100

    ## 遍历数据库找到最相近的编码
    for (name, db_enc) in database.items():
        ### 计算目标编码与当前数据库编码之间的L2差距。
        dist = np.linalg.norm(encoding - db_enc)

        ### 如果差距小于min_dist，那么就更新名字与编码到identity与min_dist中。
        if dist < min_dist:
            min_dist = dist
            identity = name

    # 判断是否在数据库中
    if min_dist > 0.7:
        print("抱歉，您的信息不在数据库中。")

    else:
        print("姓名" + str(identity) + "  差距：" + str(min_dist))

    return min_dist, identity


"""
人脸验证解决了更容易的1:1匹配问题，人脸识别解决了更难的1∶k匹配问题。

三重损失是训练神经网络学习人脸图像编码的一种有效的损失函数。

相同的编码可用于验证和识别。测量两个图像编码之间的距离可以确定它们是否是同一个人的图片。
"""