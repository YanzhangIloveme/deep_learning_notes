# 欢迎来到恐龙岛，恐龙生活于在6500万年前，现在研究人员在试着复活恐龙，
# 而你的任务就是给恐龙命名，如果一只恐龙不喜欢它的名字，它可能会狂躁不安，所以你要谨慎选择

# 你的助手已经收集了他们能够找到的所有恐龙名字，并编入了这个数据集,为了构建字符级语言模型来生成新的名称，你的模型将学习不同的名称模式，并随机生成新的名字。
# 希望这个算法能让你和你的团队远离恐龙的愤怒。

# +=============================================================
# 1 如何存储文本数据以便使用RNN进行处理。
# 2 如何合成数据，通过每次采样预测，并将其传递给下一个rnn单元。
# 3 如何构建字符级文本生成循环神经网络。
# 4 为什么梯度修剪很重要?

# 我们将首先加载我们在rnn_utils中提供的一些函数。具体地说，我们可以使用rnn_forward和rnn_backward等函数，这些函数与前面实现的函数相同。

import numpy as np
import random
import time

import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/RNN')
import cllm_utils

#%% 1.1 - 数据集与预处理
data = open("datasets/RNN/dinos.txt", "r").read()
# 转化为小写字符
data = data.lower()
# 转化为无序且不重复的元素列表
chars = list(set(data))
# 获取大小信息
data_size, vocab_size = len(data), len(chars)
print(chars)
print("共计有%d个字符，唯一字符有%d个"%(data_size,vocab_size))

# 这些字符是a-z（26个英文字符）加上“\n”（换行字符），在这里换行字符起到了在视频中类似的EOS（句子结尾）的作用
# 这里表示了名字的结束而不是句子的结尾。
# 下面我们将创建一个字典，每个字符映射到0-26的索引，然后再创建一个字典，它将该字典将每个索引映射回相应的字符字符
# 它会帮助我们找出softmax层的概率分布输出中的字符。我们来创建char_to_ix 与 ix_to_char字典。
char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}
print(char_to_ix)
print(ix_to_char)

# 1.2 模型回顾

"""
模型结构如下： 
    1 初始化参数
    2 循环：
        - 计算forward propagation
        - 计算梯度
        - 梯度修剪
        - 施行backward propagation
    3 返回学习后参数


"""

#%% 2  构建模型中的模块
# 在这部分，我们将来构建整个模型中的两个重要的模块：
#  1 梯度修剪：避免梯度爆炸
#  2 取样:一种用来产生字符的技术

"""
接下来我们将实现一个修剪函数，该函数输入一个梯度字典输出一个已经修剪过了的梯度。有很多的方法来修剪梯度，我们在这里使用一个比较简单的方法。
梯度向量的每一个元素都被限制在[ − N ， N ] [-N，N][−N，N]的范围，通俗的说，有一个maxValue（比如10），如果梯度的任何值大于10，那么它将被设置为10，
如果梯度的任何值小于-10，那么它将被设置为-10，如果它在-10与10之间，那么它将不变
"""
def clip(gradients, maxValue):
    """
     使用maxValue来修剪梯度

     参数：
         gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
         maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内

     返回：
         gradients -- 修剪后的梯度
     """
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

# test
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])

#%% 2.2 采样
# 现在我们来实现sample函数，它由以下四步构成：
# 1 网络的第一个"虚"输入x<1> = 0(零向量)
# 2 运行forward propagation，得到a<1>与y<1>
# 3 采样，用y<1> (softmax出的概率分布)选择下一个字符的索引；使用np.random.choice()函数
# 4 在sample中实现最后一步用x<t+1>的值覆盖变量x（当前储存x<t>)

def sample(parameters, char_to_is, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样

    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子

    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """
    # 从parameters 中获取参数
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    # 步骤1
    ## 创建独热向量x
    x = np.zeros((vocab_size,1))
    ## 使用0初始化a_prev
    a_prev = np.zeros((n_a,1))
    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []
    # IDX是检测换行符的标志，我们将其初始化为-1。
    idx = -1
    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    #（我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    counter = 0
    newline_character = char_to_ix["\n"]
    while (idx != newline_character and counter < 50):
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)
        # 设定随机种子
        np.random.seed(counter + seed)
        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        # 添加到索引中
        indices.append(idx)
        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        # 更新a_prev为a
        a_prev = a
        # 累加器
        seed += 1
        counter +=1

    if(counter == 50):
        indices.append(char_to_ix["\n"])
    return indices
# test
np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])

#%% 3 构建语言模型 - 3.1 - 梯度下降
#  在这里，我们将实现一个执行随机梯度下降的一个步骤的函数（带有梯度修剪）。我们将一次训练一个样本，所以优化算法将是随机梯度下降，这里是RNN的一个通用的优化循环的步骤：
"""
1 计算向前传播
2 计算反向传播梯度
3 修剪梯度
4 更新参数
"""
def rnn_forward(X, Y, a_prev, parameters):
    """
    通过RNN进行前向传播，计算交叉熵损失。

    它返回损失的值以及存储在反向传播中使用的“缓存”值。
    """
    ...
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    """
    通过时间进行反向传播，计算相对于参数的梯度损失。它还返回所有隐藏的状态
    """
    ...
    return gradients, a

def update_parameters(parameters, gradients, learning_rate):
    """
    Updates parameters using the Gradient Descent Update Rule
    """
    ...
    return parameters


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    执行训练模型的单步优化。

    参数：
        X -- 整数列表，其中每个整数映射到词汇表中的字符。
        Y -- 整数列表，与X完全相同，但向左移动了一个索引。
        a_prev -- 上一个隐藏状态
        parameters -- 字典，包含了以下参数：
                        Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b -- 偏置，维度为(n_a, 1)
                        by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate -- 模型学习的速率

    返回：
        loss -- 损失函数的值（交叉熵损失）
        gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
        a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """

    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients, 5)

    # 更新参数
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]

# test
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])

#%% 3.2 - 训练模型
# 给定恐龙名称的数据集，我们使用数据集的每一行(一个名称)作为一个训练样本。每100步随机梯度下降，你将抽样10个随机选择的名字，看看算法是怎么做的。
# 记住要打乱数据集，以便随机梯度下降以随机顺序访问样本。当examples[index]包含一个恐龙名称（String）时，为了创建一个样本（X,Y），你可以使用这个：
def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙名字

    参数：
        data -- 语料库
        ix_to_char -- 索引映射字符字典
        char_to_ix -- 字符映射索引字典
        num_iterations -- 迭代次数
        n_a -- RNN单元数量
        dino_names -- 每次迭代中采样的数量
        vocab_size -- 在文本中的唯一字符的数量

    返回：
        parameters -- 学习后了的参数
    """
    # 从vocab_size中获取n_x、n_y
    n_x, n_y = vocab_size, vocab_size
    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)
    # 初始化损失
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)
    # 构建恐龙名称列表
    with open("datasets/RNN/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)
    # 初始化LSTM隐藏状态
    a_prev = np.zeros((n_a,1))
    # 循环
    for j in range(num_iterations):
        # 定义一个训练样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        # 选择学习率为0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters

#%% test
#开始时间
import time
start_time = time.perf_counter()

#开始训练
parameters = model(data, ix_to_char, char_to_ix, num_iterations=3500)

#结束时间
end_time = time.perf_counter()

#计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")