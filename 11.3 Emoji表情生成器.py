#   在这里，我们要学习使用词向量来构建一个表情生成器。
#   你有没有想过让你的文字也有更丰富表达能力呢？比如写下“Congratulations on the promotion! Lets get coffee and talk. Love you!”，
#   那么你的表情生成器就会自动生成“Congratulations on the promotion! ? Lets get coffee and talk. ☕️ Love you! ❤️”。

#   另一方面，如果你对这些表情不感冒，而你的朋友给你发了一大堆的带表情的文字，那么你也可以使用表情生成器来怼回去。
#   我们要构建一个模型，输入的是文字（比如“Let’s go see the baseball game tonight!”），输出的是表情（⚾️）。
#   在众多的Emoji表情中，比如“❤️”代表的是“心”而不是“爱”，但是如果你使用词向量，那么你会发现即使你的训练集只明确地将几个单词与特定的表情符号相关联，
#   你的模型也了能够将测试集中的单词归纳、总结到同一个表情符号，甚至有些单词没有出现在你的训练集中也可以。
#   在这里，我们将开始构建一个使用词向量的基准模型（Emojifier-V1），然后我们会构建一个更复杂的包含了LSTM的模型（Emojifier-V2）。



import numpy as np
import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets')
from wordvector import emo_utils
import emoji
import matplotlib.pyplot as plt
#%% 1 基准模型 Emojifier-V1
# 我们来构建一个简单的分类器，首先是数据集（X，Y）：
# X：包含了127个字符串类型的短句
# Y：包含了对应短句的标签（0-4）： {0: heart, 1: baseball, 2:smile, 3:disappointed, 4:fork_and_knife}
X_train, Y_train = emo_utils.read_csv('datasets/wordvector/data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('datasets/wordvector/data/test.csv')

maxLen = len(max(X_train, key=len).split())
# test:
print(X_train[3], emo_utils.label_to_emoji(Y_train[3]))

# Emojifier-V1 结构
# -- 模型的输入是一段文字（比如“l lov you”），输出的是维度为(1,5)的向量，最后在argmax层找寻最大可能性的输出。
# -- 现在我们将我们的标签Y转换成softmax分类器所需要的格式，从(m,1) -> (m,5)
# -- 第一步就是把输入的句子转换为词向量，然后获取均值，我们依然使用50维的词嵌入，现在我们加载词嵌入：
Y_oh_train = emo_utils.convert_to_one_hot(Y_train, C=5)
Y_oh_test = emo_utils.convert_to_one_hot(Y_test, C=5)
print("{0}对应的独热编码是{1}".format(Y_train[3], Y_oh_train[3]))
#%% 将输入句子转化为词向量
word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('datasets/wordvector/data/glove.6B.50d.txt')
# 把每个句子转换为小写，然后分割为列表。我们可以使用X.lower() 与 X.split()。
# 对于句子中的每一个单词，转换为GloVe向量，然后对它们取平均。

def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。

    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec_map -- 字典类型，单词映射到50维的向量的字典

    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """
    # step1 : 分割句子，转换为列表。
    words = sentence.lower().split()
    # step2 : 初始化均值词向量
    avg = np.zeros(50,)
    # step3 : 对词向量取平均。
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg, len(words))
    return avg

avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = ", avg)

#%% 我们现在应该实现所有的模型结构了，在使用sentence_to_avg()之后，进行前向传播，计算损失，再进行反向传播，最后再更新参数。
def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。

    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。

    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)
    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50
    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    # 将Y转换成独热编码
    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)
    # 优化循环
    for t in range(num_iterations):
        for i in range(m):
            # 获取第i个训练样本的均值
            avg = sentence_to_avg(X[i], word_to_vec_map)
            # 前向传播
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)
            # 计算第i个训练的损失
            cost = -np.sum(Y_oh[i]*np.log(a))
            # 计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t,cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)
    return pred, W, b

# 训练
pred, W, b = model(X_train, Y_train, word_to_vec_map)
# test
print("=====训练集====")
pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
print("=====测试集====")
pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)
#%% 思考
#   假设有5个类别，随机猜测的准确率在20%左右，但是仅仅经过127个样本的训练，就有很好的表现。在训练集中，算法看到了“I love you”的句子，其标签为“❤️”，
#   在训练集中没有“adore”这个词汇，如果我们写“I adore you”会发生什么？
X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = emo_utils.predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
emo_utils.print_predictions(X_my_sentences, pred)

"""
因为词嵌入的原因，“adore”与“love”很相似，所以它可以正确表达出“❤️”，但是在“you are not happy”中却表达了“❤️”，其原因是我们这个算法使用均值，忽略了排序，
所以不善于理解“not happy”这一类词汇。
我们把矩阵打印出来应该会帮助你理解哪些类让模型学习起来比较困难，横轴为预测，竖轴为实际标签。
"""
print(" \t {0} \t {1} \t {2} \t {3} \t {4}".format(emo_utils.label_to_emoji(0), emo_utils.label_to_emoji(1), \
                                                 emo_utils.label_to_emoji(2), emo_utils.label_to_emoji(3), \
                                                 emo_utils.label_to_emoji(4)))
import pandas as pd
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
emo_utils.plot_confusion_matrix(Y_test, pred_test)
plt.show()

"""
结论：

1 即使你只有128个训练样本，你也可以得到很好地表情符号模型，因为词向量是训练好了的，它会给你一个较好的概括能力。

2 Emojifier-V1是有缺陷的，比如它不会把“This movie is not good and not enjoyable”划分为不好一类，因为它只是将所有单词的向量做了平均，没有关心过顺序。

"""
# do my own
X_my_sentences =  np.array(['you motherfucker'])
Y_my_labels = np.array([[0]])

pred = emo_utils.predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
emo_utils.label_to_emoji(int(pred[0][0]))
#%% 2  - Emojifier-V2：在Keras中使用LSTM模块
# 现在我们构建一个能够接受输入文字序列的模型，这个模型会考虑到文字的顺序。Emojifier-V2依然会使用已经训练好的词嵌入。

import numpy as np
np.random.seed(0)
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

np.random.seed(1)
from tensorflow.keras.initializers import glorot_uniform

"""
模型结构如下：

 my(x1)  grandfather(x2)  cooks(x3) ... meals(xt)
                -- embedding -- 
 LSTM -> LSTM -> LSTM ->  ... -> LSTM
                -- Dropout -- 
 LSTM -> LSTM -> LSTM ->  ... -> LSTM
                                  |
                                output
                                  |
                                softmax

在这个部分中，我们会使用mini-batches来训练Keras模型，但是大部分深度学习框架需要使用相同的长度的文字，这是因为如果你使用3个单词与4个单词的句子，
那么转化为向量之后，计算步骤就有所不同（一个是需要3个LSTM，另一个需要4个LSTM），所以我们不可能对这些句子进行同时训练。

那么通用的解决方案是使用填充。指定最长句子的长度，然后对其他句子进行填充到相同长度。比如：指定最大的句子的长度为20，我们可以对每个句子使用“0”来填充，
直到句子长度为20
"""
#%% 嵌入层 The Embedding layer
#   在keras里面，嵌入矩阵被表示为“layer”，并将正整数（对应单词的索引）映射到固定大小的Dense向量（词嵌入向量），它可以使用训练好的词嵌入来接着训练或者直接初始化。
#   在这里，我们将学习如何在Keras中创建一个Embedding()层，然后使用Glove的50维向量来初始化。因为我们的数据集很小，所以我们不会更新词嵌入，而是会保留词嵌入的值。
#   在Embedding()层中，输入一个整数矩阵（batch的大小，最大的输入长度），我们可以看看下图：
"""
i love you
【185457,226278,394475,0,0】 -> embedding layers -> 【e185457, e226278 ..】
这个例子展示了两个样本通过embedding层，两个样本都经过了`max_len=5`的填充处理，最终的维度就变成了`(2, max_len, 5)`，这是因为使用了50维的词嵌入。

输入的最大的数（也就是说单词索引）不应该超过词汇表包含词汇的数量，这一层的输出的数组的维度为(batch size, max input length, dimension of word vectors)。
第一步就是把所有的要训练的句子转换成索引列表，然后对这些列表使用0填充，直到列表长度为最长句子的长度。
我们先来实现一个函数，输入的是X（字符串类型的句子的数组），再转化为对应的句子列表，输出的是能够让Embedding()函数接受的列表或矩阵（参见图2-4）。
"""
def sentences_to_indices(X, word_to_index, max_len):
    """
    输入的是X（字符串类型的句子的数组），再转化为对应的句子列表，
    输出的是能够让Embedding()函数接受的列表或矩阵（参见图4）。

    参数：
        X -- 句子数组，维度为(m, 1)
        word_to_index -- 字典类型的单词到索引的映射
        max_len -- 最大句子的长度，数据集中所有的句子的长度都不会超过它。

    返回：
        X_indices -- 对应于X中的单词索引数组，维度为(m, max_len)
    """
    m = X.shape[0]  # 训练集数量
    # 使用0初始化X_indices
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        # 将第i个居住转化为小写并按单词分开。
        sentences_words = X[i].lower().split()
        # 初始化j为0
        j = 0
        # 遍历这个单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices

# test
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)

#%%  现在我们就在Keras中构建Embedding()层
# 我们使用的是已经训练好了的词向量，在构建之后，使用sentences_to_indices()生成的数据作为输入，Embedding()层将返回每个句子的词嵌入。
# 我们现在就实现pretrained_embedding_layer()函数，它可以分为以下几个步骤：
# - 1 使用0来初始化嵌入矩阵。
# - 2 使用word_to_vec_map来将词嵌入矩阵填充进嵌入矩阵。
# - 3 在Keras中定义嵌入层，当调用Embedding()的时候需要让这一层的参数不能被训练，所以我们可以设置trainable=False。
# - 4 将词嵌入的权值设置为词嵌入的值
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建Keras Embedding()层，加载已经训练好了的50维GloVe向量

    参数：
        word_to_vec_map -- 字典类型的单词与词嵌入的映射
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        embedding_layer() -- 训练好了的Keras的实体层。
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    # 初始化嵌入矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))
    # 将嵌入矩阵的每行的“index”设置为词汇“index”的词向量表示
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    # 定义Keras的embbeding层
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    # 构建embedding层。
    embedding_layer.build((None,))
    # 将嵌入层的权重设置为嵌入矩阵。
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

#%% 构建Emojifier-V2

# 现在实现Emojifier_V2()函数，模型的输入是(m, max_len)，定义在了input_shape中，输出是(m, C=5)，
def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    实现Emojify-V2模型的计算图

    参数：
        input_shape -- 输入的维度，通常是(max_len,)
        word_to_vec_map -- 字典类型的单词与词嵌入的映射。
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        model -- Keras模型实体
    """
    # 定义sentence_indices为计算图的输入，维度为(input_shape,)，类型为dtype 'int32'
    sentence_indices = Input(input_shape, dtype='int32')
    # 创建embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # 通过嵌入层传播sentence_indices，你会得到嵌入的结果
    embeddings = embedding_layer(sentence_indices)
    # 通过带有128维隐藏状态的LSTM层传播嵌入
    # 需要注意的是，返回的输出应该是一批序列。
    X = LSTM(128, return_sequences=True)(embeddings)
    # 使用dropout，概率为0.5
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    # 使用dropout，概率为0.5
    X = Dropout(0.5)(X)
    # 通过softmax激活的Dense层传播X，得到一批5维向量。
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model

#   因为数据集中所有句子都小于10个单词，所以我们选择max_len=10。在接下来的代码中，你应该可以看到有“20,223,927”个参数，
#   其中“20,000,050”个参数没有被训练（这是因为它是词向量），剩下的是有“223,877”被训练了的。因为我们的单词表有400,001个单词，所以是
#   400,001∗50=20,000,050个不可训练的参数
max_Len = 10
model = Emojify_V2((max_Len,), word_to_vec_map, word_to_index)
model.summary()

#%%   与往常一样，在Keras中创建模型以后，我们需要编译并评估这个模型。我们可以使用categorical_crossentropy 损失, adam 优化器与 [‘accuracy’] 指标。
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#   现在我们开始训练模型，Emojifier-V2模型是以(m, max_len)为输入，(m, number of classes)为输出。
#   我们需要将X_train转化为X_train_indices，Y_train转化为Y_train_oh。

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C = 5)
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

# test
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)

print("Test accuracy = ", acc)

#%% test my own

x_test = np.array(['I want to fuck you'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  emo_utils.label_to_emoji(np.argmax(model.predict(X_test_indices))))

"""
在以前的Emojiy-V1模型中它不会正确标记“不开心”，但是我们在Emojiy-V2中纠正了它。目前的Emojiy-V2模型在理解否定词上依旧是不大稳定的，
这是因为训练集比较小，如果训练集比较大的话LSTM就会表现的更好。

"""