#%% 1 mini-batch梯度下降

# 核心思想： 使用X原始数据集的子集，避免全量遍历数据集再进行梯度下降；

# 比如 X = [x[1], x[2], ... , x[m]]每1000层切分一次；
# x{1} = [x[1], x[2],...x[1000]];
# x{2} = [x[1001], x[1002],...x[2000]]

# mini-batch的损失方程并不是持续下降；而是波动性向下；
# if mini-batch size = m : Batch gradient descend （too long）
# if mini-batch size = 1 : Stochastic gradient descend; every example is mini-batch(noisy, 失去vector向量化加速)

# if small training set: use batch gradient descent
# typical mini-batch size: 64, 128, 256, 512 (考虑到内存，这样设置会快一些)

#%% [补] 指数加权平均（是一些比梯度下降更快的优化算法的基础）

"""
以伦敦气温为例引入：

v0 = 0
v1 = 0.9*v0 + 0.1 theta[1]
v2 = 0.9*v1 + 0.1 theta[2]
...
vt = 0.9*vt-1 + 0.1 theta[t]

==============================
公式：
V[t] = beta * V[t-1] + (1-beta) * theta[t]
所以V[t]大约是 1/(1-beta) days的每日温度 所以如果beta是0.9，则是10天的平均

==============================
指数加权平均的偏差修正（指数甲醛前期预测不好；）
使用 V[t]/(1-beta^t) 代替 V[t]
随着t增加，修正指数下降；

"""

#%% 2 Momentum 动量梯度下降法
# 核心思想是计算梯度的指数加权平均并使用这个更新权重

"""
On iteration T:
compute dW, dB on mini-batch
vdW = beta*vdW + (1-beta)*dW
vdb = beta*vdb + (1-beta)*db
w := w-alpha * vdw
b := b-alpha * vdb

其核心思想是降低纵向的波动，使用指数平均将波动降低，这样可以使用更大的学习率增快在横向优化速度
通常beta 为0.9， 即考虑过去10个的平均；一般不用偏差修正，因为迭代10此以后基本消除偏差影响

另外一个版本是删除(1-beta)这个系数
vdW = beta*vdW + dW
vdb = beta*vdb + db
其影响是vdW缩小(1-beta)倍，即除以了（1-beta）
但是需要对alpha学习率调整，所以通常用上一个版本

"""

#%% 3 RMSprop 优化算法  (root mean square prop) --> 可以加速梯度下降

"""
RMSprop核心思想是在参数空间更平缓的方向，让梯度取得更大的进步
由于该方向平缓，所以历史梯度平方和较小，对应的学习下降幅度较小，所以我们除以一个小的数修正；
同时，在陡峭方向由于波动过大，我们除以一个大数，使其变得平缓，从而加快训练速度

sdW = beta2 * sdW + (1-beta2) * dW ** 2 
sdb = beta2 * sdb + (1-beta2) * db ** 2

w := w- alpha * dw / (np.sqrt(sdW+epslion))  
b := b- alpha * db / (np.sqrt(sdb+epslion))  
where epslion is a small num (usually 1e-8) for removing divide-zero errors

Here we use beta2 instead of beta. That's because a new algorithm which combine RMSprop and Momentum GD usually 
perform better in practice. 
"""

#%% 4 Adam 优化算法： Adaptive Moment Estimation
"""
初始化参数： vdW = 0 | sdW = 0 | vdb = 0 | sdb = 0

on iteration t: compute dW, db by using mini-batch

- 使用momentum指数加权平均数： 
vdw = beta1 * vdW + (1-beta1) * dW
vdb = beta1 * dvb + (1-beta1) * db

- 使用RMSprop更新：
sdW = beta2 * sdW + (1-beta2) * dW ** 2 
sdb = beta2 * sdb + (1-beta2) * db ** 2

- 计算偏差修正

vdW_C = vdW /(1 - beta1^t), vdb_C = vdb /(1 - beta1^t) where t stands for iterations
sdW_C = sdW /(1 - beta2^t), sdb_C = sdb /(1 - beta2^t) where t stands for iterations

W := W - alpha * vdW_C / (spqt(sdw_c + eplsion))
b := b - alpha * vdb_C / (spqt(sdb_c + eplsion))


- 超参数： 
learning rate alpha: to be turn
beta1: 0.9 (dW)
beta2: 0.999 (dW**2)
eplsion: 1e-8 (不重要)

"""

#%% 5 学习率衰减

"""
由于bach里面存在噪音，在收敛阶段我们希望学习率下降，以供最后收敛
我们可以记住迭代的轮数，完整走完一遍数据集为一个epoch

【学习率递减公式】
alpha = 1/ (1 + decay_rate * epoch_num) * alpha[0]
【指数学习率递减公式】：
alpha = 0.95^(epoch-num) * alpha[0]  
【常数递减学习率1】
alpha = k/sqrt(epoch_num) * alpha[0]
【常数递减学习率2】
alpha = k/sqrt(t) * alpha[0] where t stands for mini-batch 的数字t
【离散步骤的学习率】
设定一个规则，当触发时候修正学习率

"""

#%% 6 局部最优问题

"""
Unlikely to get stuck in a bad local optima (在大数据集和大规模参数条件下)
Plateaus can make learning slow
"""
