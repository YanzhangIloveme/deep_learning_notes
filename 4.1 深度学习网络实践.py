#%% 1 偏差/方差

"""
High bias -> Bigger network / train longer /other NN architecture
High var  -> More data / Regularization /
"""

#%% 2 正则化
"""
l2: min J(w,b) = 1/m * sum(L(yhat[i], y[i])) + lambda/2m * W.T*W
l1: min J(w,b) = 1/m * sum(L(yhat[i], y[i])) + lambda/2m * np.sum(np.abs(W))

L2正则化效果：权重缩减

dW[l] = (from backpop) + lambda / m * W[l]
W[l] := W[l] - alpha * dW[l] = W[l] - alpha * ((from backpop) + lambda / m * W[l]) 
      = (1 - alpha* lambda / m ) * W[l] - alpha * (from backpop)

由于W[l]前乘上了一个小于1的系数，导致L2正则化后的系数衰减；

当正则化参数lambda 很大当时候，W趋于0，以tanh激活函数为例子， Z[l] = W[l] * A[l-1] + b[l] 
此时Z比较小，激活函数tanh在此区间接近线性。多重的线性组合并不能提升模型复杂程度。

"""

#%% 3 dropout

"""
dropout原理很简单，实现方式引入向量d[l], 引入保存的概率keep_prob
在L层引入drop的时候：
d[l] = np.random.rand(A[l].shape[0],A[l].shape[1]) < keep_prob
d[l] 是 [True/False]向量，python中对应1和0，1为保留，0为去掉；
A[l] = np.multiply(A[l], d[l]) 元素两两相乘，则留下的就是好的；
最后，为了避免drop掉的影响，保持A[l]的期望不变，则令A[l] /= keep_prob
如Z[4] = W[4] * A[3] + b[4], A[3]/=keep_prob;


做预测的时候，不使用drop out


#原理： make nn cannt rely on any one feature, so have to spread out weights;
"""
import numpy as np
np.random.rand(3,3)<0.8

#%% 4 其他正则化方法

# 1) data augmentation: 对图片等数据进行变化，如对称，旋转，扭曲等；
# 2) early stop：随着iterations增加，test在部分iterations最优。
#       -- 这个方法优点在于不需要迭代超参数；
#       -- 缺点在于无法正交化（orthogonalization）它必须在bias和variance同时解决，不能独立解决其中一个问题；


#%% 5 Normalizing input 标准化数据后，可以提升训练速度

# Step 1: 零均值化 u = 1/m * sum(x[i]), x:=x-u
# Step 2: normalize var sigma^2， x/sigma^2

# 注意：当使用u和sigma标准化train数据集对时候，用同样的u和sigma标准化test数据集，否则维度不一致

# 原理：如果W没有标准化，W1对取之是1->1000， W2是0->1，那么损失函数非常不均衡；可能需要一个很小的学习率（受w2影响）
# 但是标准化后，w1和w2在同一个维度，损失函数更加均匀，这样优化更加简单

#%% 6 梯度消失和梯度爆炸

"""
设定：
g[z] = z
y = W[l]*W[l-1]*W[l-2]...*W[1] * X
当W[l] = [[1.5,0],[0,1.5]] 多个W的叠加，[[1.5^L,0],[0,1.5^L]] 
当W[l] = [[.5,0],[0,.5]] 多个W的叠加，[[.5^L,0],[0,.5^L]]

所以在深度的网络的时候，W会随着指数层级变动；
可以证明，其梯度也随之指数级增长/坍缩


具体：如果一个四层全联接网络，dW[2]=dL/dA[4] * dA[4]/dA[3] * dA[3]/dA[2] * dA[2]/dw[2]
所以，之前是对激活函数求导，层数多的时候，梯度以指数形式增加，发生爆炸，小于1，指数形式衰减，梯度消失

网络接近输出对层学习很好，但是接近输入对层学习很慢；这个就是反向传播的失败；Hinton提出capsule就是为了彻底摒弃反向传播
=================
从激活函数角度：
sigmoid的梯度不可能超过0.25, 梯度最大出在0所实现，是(exp(-x)/(1+exp(-x)^2)) -> 0.25
tanh的梯度：比sigmoid好一些，但是导数仍然是小于1
=================
解决方案：

1 预训练+微调：Hinton的一篇论文，提出用无监督逐层训练。

其基本思想是每次训练一层隐节点，训练时将上一层隐节点的输出作为输入，而本层隐节点的输出作为下一层隐节点的输入，此过程就是逐层“预训练”（pre-training）
在预训练完成后，再对整个网络进行“微调”（fine-tunning）
Hinton在训练深度信念网络（Deep Belief Networks中，使用了这个方法，在各层预训练完成后，再利用BP算法对整个网络进行训练。
此思想相当于是先寻找局部最优，然后整合起来寻找全局最优，此方法有一定的好处，但是目前应用的不是很多了。

2 梯度剪切、正则
梯度剪切这个方案主要是针对梯度爆炸提出的，其思想是设置一个梯度剪切阈值，然后更新梯度的时候，如果梯度超过这个阈值，那么就将其强制限制在这个范围之内。
这可以防止梯度爆炸。

另外一种解决梯度爆炸的手段是采用权重正则化（weithts regularization）比较常见的是l1正则，和l 2 l2l2正则。如果发生梯度爆炸，权值的范数就会变的非常大，
通过正则化项，可以部分限制梯度爆炸的发生。 （但是往往梯度消失出现的更多一些）

3 relu、leakrelu、elu等激活函数

【relu】的梯度为1！
 -- 解决了梯度消失、爆炸的问题
 -- 计算方便，计算速度快
 -- 加速了网络的训练
 == 由于负数部分恒为0，会导致一些神经元无法激活（可通过设置小学习率部分解决）
 == 输出不是以0为中心的
 
【leakrelu】leakrelu = max(k*x, x), k一般用0.01/0.02或者学习而来，解决了0区间的问题，包含relu所有优点
【elu】elu = x if x>0, a*(exp(x) -1), otherwise 解决relu0区间问题，比leakrelu计算耗费时间

4 Batchnorm：具有加速网络收敛速度，提升训练稳定性的效果
 -- Batchnorm本质上是解决反向传播过程中的梯度问题。batchnorm全名是batch normalization，简称BN，即批规范化，通过规范化操作将
    输出信号x规范化保证网络的稳定性。之后具体讲到了再学习

5 LSTM结构：是不那么容易发生梯度消失的，主要原因在于LSTM内部复杂的“门”(gates)，，LSTM通过它内部的“门”可以接下来更新的时候“记住”前几次训练的”残留记忆“，
  因此，经常用于生成文本中。目前也有基于CNN的LSTM，感兴趣的可以尝试一下。
 
 
 
 
"""

#%% 7 nn权重初始化

"""

Z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn
所以n越大，我们期望w越小，其中一种做法，是设置 Var(w[i]) = 1/n， 即保证输入输出符合同一个分布
==========
var(w[i]*x[i]) = E[w[i]]^2 * var(x[i]) + E[x[i]]^2 * var(w[i]) + var(w[i])*var(x[i])
当我们假设输入和权重都是0均值时（目前有了BN之后，这一点也较容易满足），上式可以简化为：
var(w[i]*x[i]) = var(wi)var(xi)
进一步假设：输入x和权重w独立同分布则有：var(y) = ni * var(w[i]) * var(x[i])
var(w[i]) = 1/ni，此极为Xavier的方法；



W[l] = np.random.randn(shape[0],shape[1]) * np.sqrt(1/n[l-1])
* 当激活函数是relu当时候，使用Var(w[i]) = 2/n 效果更好

如果激活函数当输入特征被0均值，标准方差（1）, Z也会被映射到相应的维度，这样W在刚刚好的位置，一定程度上避免了梯度爆炸/消失

不同激活函数对应的初始值；
[Tanh] -> np.sqrt(1/n[l-1])   -> Xavier初始化
       -> np.sqrt(2/(n[l-1]+n[l])) -> Yoshua Bengio的初始化方法
[relu] -> np.sqrt(2/n[l-1])

更一般的，可以给var添加一个超参数，可以tune的。
"""

# 详解机器学习中的梯度消失、爆炸原因及其解决方法
# 参考阅读文献：https://blog.csdn.net/qq_25737169/article/details/78847691

#%% 8 梯度数值逼近
"""
为了保证backpop准确，有一个梯度检验确保准确，为了保证这个我们需要首先学习如何对梯度进行逼近；
设有一个单调函数，f，有一个小整数epsilon，根据导数定义，epsilon趋紧0时候，导数是[f(x+epsilon)-f(x-epsilon)]/2epsilon
如设epsilon为0.01，对f(x)=x^3, x=1时候，g(x) = 3.001
(用单边的化不准确，用双边更准确，但是对极限来说，两个没有区别)
"""

#%% 9 梯度检验
"""
梯度检验来验证backprop非常高效率

1: take W[1], b[1],...,W[l], b[l] and reshape into a big vector theta, 所以J(w[l],b[l]..) = J(theta)
2: take dW[1],db[1],...dW[l],db[l] and reshape into a big vector d(theta)

Gradient Checking ：
for each i:
    d(theta[i])_approximate = [J(theta[1],theta[2],...,theta[i]+epsilon,...) - J(theta[1],theta[2],...,theta[i],...)]/(2*epsilon)
    --> approximately equ = d(theta[i]) = dJ/d(theta[i])

之后，检查 d(theta[i])_approximate和d(theta[i])是否接近，可以用标准化后欧式距离来计算。
    distance =[d(theta[i])_approximate - d(theta[i])]/[d(theta[i])^2 + d(theta[i])_approximate^2] < epsilon （如1e-7）
    则说明导数逼近比较准确；
    如果比1e-3大很多，肯定有问题了
"""

#%% 10 梯度检验tips

"""
- dont use in training -only to debug
- if algo fails grad check, lock at components to try to identify bug
- remember regularization
- dont work with dropout
- run at random initialization; perhaps again after some training
"""

