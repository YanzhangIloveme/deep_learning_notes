#%% 1 一步一步搭建RNN
"""
声明：
上标[l]表示第l层：（a[4]表示第4层的激活值，W[5]与b[5]是第5层的参数）
上标(i)表示第i个样本
上标<t>表示第t个时间步
下表i表示向量的第ith项的激活值
"""
import numpy as np
import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/RNN')
import rnn_utils


# 实现RNN的一个时间步所需要计算的东西。
# 在Tx时间步上实现一个循环，以便一次处理所有输入。
#%% 1.1 RNN单元
# 我们将向量化m mm个样本，因此，x<t>的维度为(nx, m), a<t>的维度为(na, m)
# a<t> = g(Waa * a<t-1> + Wax * x<t> + ba) 通常应用tanh，也有用relu的
# y<t> = g(Wya * a<t> + by>
def rnn_cell_forward(xt, a_prev, parameters):
    """
    参数：
        xt -- 时间步“t”输入的数据，维度为（n_x, m）
        a_prev -- 时间步“t - 1”的隐藏隐藏状态，维度为（n_a, m）
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a_next -- 下一个隐藏状态，维度为（n_a， m）
        yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
        cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba  = parameters["ba"]
    by  = parameters["by"]
    # 使用上面的公式计算下一个激活值
    a_next = np.tanh(np.dot(Waa, a_prev)+np.dot(Wax, xt)+ba)
    # 使用上面的公式计算当前单元的输出 # y<t> = g(Wya * a<t> + by>
    yt_pred = rnn_utils.softmax(np.dot(Wya,a_next)+by)
    # 保存反向传播需要的值
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache

# 测试一下
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)

#%% 1.2 RNN的前向传播
# 可以看到的是RNN是刚刚构建的单元格的重复连接，如果输入的数据序列经过10个时间步，那么将复制RNN单元10次
# 每个单元将前一个单元中的隐藏状态a<t-1>和当前时间步的输入数据x<t>作为输入，它为时间步输出隐藏状态a<t>和预测y<t>
# 我们要实现前向传播的代码，它由以下几步构成：
# 1 创建0向量zeros(a),它将保存RNN计算的所有的隐藏状态。
# 2 使用a0初始化"next"隐藏状态
# 3 循环所有时间步：（1）使用rnn_cell_forward函数来更新“next”隐藏状态与cache。 （2）使用a aa来保存“next”隐藏状态（第t tt）个位置。
#                （3）使用y来保存预测值。 （4）把cache保存到“caches”列表中。

def rnn_forward(x, a0, parameters):
    """
    参数：
        x -- 输入的全部数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为 (n_a, m)
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """
    # 初始化“caches”，它将以列表类型包含所有的cache
    caches = []
    # 获取 x 与 Wya 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    # 使用0来初始化“a” 与“y”
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    # 初始化“next”
    a_prev = a0
    # 遍历所有时间步
    for t in range(T_x):
        ## 1.使用rnn_cell_forward函数来更新“next”隐藏状态与cache。
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_prev, parameters)
        ## 2.使用 a 来保存“next”隐藏状态（第 t ）个位置。
        a[:, :, t] = a_next
        ## 3.使用 y 来保存预测值。
        y_pred[:, :, t] = yt_pred
        ## 4.把cache保存到“caches”列表中。
        caches.append(cache)
        ## 5 将a_prev更改为a_next进行下一轮输入
        a_prev = a_next

    caches = (caches, x)

    return a, y_pred, caches

# 测试一下

np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))

# 我们构建了循环神经网络的前向传播函数，这对于某些应用程序来说已经足够好了，但是它还存在梯度消失的问题
# 当每个输出y<t>是根据局部上下文预测的时候，他的效果是比较好的；但是较长的话就不行了；

#%% 2 LSTM 网络

"""
C~<t> = tanh(Wc * [a<t-1>, x<t>] + bc) 【更新单元】
gammaU = sigmoid(Wu[a<t-1>,x<t>]+bu) 【更新门】
gammaF = sigmoid(Wf[a<t-1>,x<t>]+bf) 【遗忘门】
gammaO = sigmoid(Wo[a<t-1>,x<t>]+bo) 【输出门】
C<t> = gammaU * C~<t> + gammaF * C<t-1>
a<t> = gammaO * tanh(C<t>)

"""

#%% 2.1 LSTM-CELL
"""
步骤：
1. 把a<t-1>, x<t> 连接成一个矩阵concat=[a<t-1,x<t>].T
2. 输入公式计算多个门
3. 预测y<t>
"""


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    参数：
        xt -- 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters -- 字典类型的变量，包含了：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wu -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bu -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    返回：
        a_next -- 下一个隐藏状态，维度为(n_a, m)
        c_next -- 下一个记忆状态，维度为(n_a, m)
        yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
        cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)

    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    # 获取 xt 与 Wy 的维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    # 1.连接 a_prev 与 xt
    concat = np.zeros([n_a+n_x,m])
    concat[:n_a,:]=a_prev
    concat[n_a:,:]=xt
    # 2.根据公式计算ft、ut、c~t、c_next、ot、a_next
    ## 遗忘门，公式1
    ft = rnn_utils.sigmoid(np.dot(Wf, concat)+bf)
    ## 更新门，公式2
    ut = rnn_utils.sigmoid(np.dot(Wu, concat)+bu)
    ## 更新单元，公式3
    cct = np.tanh(np.dot(Wc, concat) + bc)
    ## 输出门，公式5
    ot = rnn_utils.sigmoid(np.dot(Wo, concat)+bo)
    ## 更新单元，公式4
    c_next = ft * c_prev + ut * cct
    ## 输出门，公式6
    a_next = ot * np.tanh(c_next)
    # 3.计算LSTM单元的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)
    # 保存包含了反向传播所需要的参数
    cache = (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache

# test
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))

#%% 2.2 LSTM向前传播
def lstm_forward(x, a0, parameters):
    """
    参数：
        x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为(n_a, m)
        parameters -- python字典，包含了以下参数：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wu -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bu -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y -- 所有时间步的预测值，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """
    caches = []
    # 获取 xt 与 Wy 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape
    # 使用0来初始化“a”、“c”、“y”
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])
    # 初始化“a_next”、“c_next”
    a_next = a0
    c_next = np.zeros([n_a, m])

    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

        # 保存新的下一个隐藏状态到变量a中
        a[:, :, t] = a_next

        # 保存预测值到变量y中
        y[:, :, t] = yt_pred

        # 保存下一个单元状态到变量c中
        c[:, :, t] = c_next

        # 把cache添加到caches中
        caches.append(cache)

    # 保存反向传播需要的参数
    caches = (caches, x)
    return a, y, c, caches
# test
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))

#%% 3 循环神经网络的反向传播
"""
tanh的导数是1-tanh(x)^2
# a<t> = g(Waa * a<t-1> + Wax * x<t> + ba) 
da<t>/dWax = [1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2] * x<t>.T
da<t>/dWaa = [1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2] * a<t-1>.T
da<t>/dba = sum([1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2])   # 注：此时db是一个数值，无法像上面weights做矩阵乘法
da<t>/dx<t> = Wax.T * [1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2] 
da<t>/da<t-1> = Waa.T * [1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2] 

# y<t> = g(Wya * a<t> + by)
"""

def rnn_cell_backward(da_next, cache):
    """
    实现基本的RNN单元的单步反向传播

    参数：
        da_next -- 关于下一个隐藏状态的损失的梯度。
        cache -- 字典类型，rnn_step_forward()的输出

    返回：
        gradients -- 字典，包含了以下参数：
                        dx -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                        dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                        dba -- 偏置向量的梯度，维度为(n_a, 1)
    """
    # 获取cache 的值
    a_next, a_prev, xt, parameters = cache
    # 从 parameters 中获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    # 计算tanh相对于a_next的梯度.
    # [1-tanh(Waa * a<t-1> + Wax * x<t> + ba)^2]
    dtanh = (1 - np.square(a_next)) * da_next
    # 计算关于Wax损失的梯度
    dxt = np.dot(Wax.T,dtanh)
    dWax = np.dot(dtanh, xt.T)
    # 计算关于Waa损失的梯度
    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    # 计算关于b损失的梯度
    dba = np.sum(dtanh, keepdims=True, axis=-1)
    # 保存这些梯度到字典内
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients
# test
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

da_next = np.random.randn(5,10)
gradients = rnn_cell_backward(da_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)


#%% 对RNN进行梯度下降

def rnn_backward(da, caches):
    """
    在整个输入数据序列上实现RNN的反向传播

    参数：
        da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x)
        caches -- 包含向前传播的信息的元组

    返回：
        gradients -- 包含了梯度的字典：
                        dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                        da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                        dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                        dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                        dba -- 关于偏置的梯度，维度为(n_a, 1)
    """

    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    a1, a0, x1, parameters = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 计算时间步“t”时的梯度
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        #从梯度中获取导数
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # 通过在时间步t添加它们的导数来增加关于全局导数的参数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    #将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
    da0 = da_prevt
    #保存这些梯度到字典内
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    return gradients

# test
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)

#%% 4 LSTM反向传播
"""
更复杂一些，但是不难，原理类似；门的导数公式如下：
dot = da_next * tanh*c_next * ot * (1-ot)
dcct = dc_next * fu*(1-tanh(c_next)^2) * fu *da_next * cct * (10tanh(cct)^2)
dut = dc_next * cct + ot*(1-tanh(c_next)^2)*cct * da_next * ut * (1-ut)
dft = dc_next * c_prev + ft*(1-tanh(c_next)^2)*c_prev*da_next*ft*(1-ft)

# 参数的导数
dWf = dft * (a_prev, xt).T
dWu = dut * (a_prev, xt).T
dWc = dcct * (a_prev, xt).T
dWo = dot * (a_prev, xt).T
# 计算db的时候，需要对axis=1求和，需要keep_dims=True
da_prev = Wf.T * dft + Wu.T * dut + Wc.T * dcct + Wo.T * dot
dc_prev = dc_next * ft + ot*(1-tanh(c_next)&+^2) * ft * da_next
dx = Wf.T * dft + Wu.T * dut + Wc.T * dcct + Wo.t * dot

"""


def lstm_cell_backward(da_next, dc_next, cache):
    """
    实现LSTM的单步反向传播

    参数：
        da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
        dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
        cache -- 来自前向传播的一些参数

    返回：
        gradients -- 包含了梯度信息的字典：
                        dxt -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWu -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbu -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
    """
    # 从cache中获取信息
    (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt, parameters) = cache

    # 获取xt与a_next的维度信息
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # 根据公式7-10来计算门的导数
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * ut + ot * (1 - np.square(np.tanh(c_next))) * ut * da_next) * (1 - np.square(cct))
    dut = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * ut * (1 - ut)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    # 根据公式11-14计算参数的导数
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWu = np.dot(dut, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbu = np.sum(dut, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # 使用公式15-17计算洗起来了隐藏状态、先前记忆状态、输入的导数。
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wu"][:, :n_a].T, dut) + np.dot(parameters["Wo"][:, :n_a].T, dot)

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wu"][:, n_a:].T, dut) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWu": dWu, "dbu": dbu,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

da_next = np.random.randn(5,10)
dc_next = np.random.randn(5,10)
gradients = lstm_cell_backward(da_next, dc_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWu\"][1][2] =", gradients["dWu"][1][2])
print("gradients[\"dWu\"].shape =", gradients["dWu"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbu\"][4] =", gradients["dbu"][4])
print("gradients[\"dbu\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

#%% LSTM网络的反向传播

def lstm_backward(da, caches):
    """
    实现LSTM网络的反向传播

    参数：
        da -- 关于隐藏状态的梯度，维度为(n_a, m, T_x)
        cachses -- 前向传播保存的信息

    返回：
        gradients -- 包含了梯度信息的字典：
                        dx -- 输入数据的梯度，维度为(n_x, m，T_x)
                        da0 -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWu -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbu -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)

    """

    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWu = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbu = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 使用lstm_cell_backward函数计算所有梯度
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # 保存相关参数
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWu = dWu + gradients['dWu']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbu = dbu + gradients['dbu']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    da0 = gradients['da_prev']

    # 保存所有梯度到字典变量内
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWu": dWu, "dbu": dbu,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

# test

np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWu\"][1][2] =", gradients["dWu"][1][2])
print("gradients[\"dWu\"].shape =", gradients["dWu"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbu\"][4] =", gradients["dbu"][4])
print("gradients[\"dbu\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)