#%% 1 深度学习参数调试处理

"""
参数列表
alpha
beta (beta1, beta2, eplsion)
layers
hadden units
learning rate decay
mini-batch size
"""

# 使用随即搜索而非网格搜索；使用从粗糙到精细的搜索策略

#%% 2 标准化激活函数-batch-normal

"""
batch 归一化使模型参数调节更robust，使训练速度加快
在实践中我们通常normalize Z[l] 而不是A[l]
公式 for l层：
mean = 1/m * sum(Z[i]) 
sigma^2 = 1/m * sum(Z[i] - mean)^2
ZN[i] = (Z[i] - mean) / sqrt(sigma^2 + eplsion)
Ztilde[i] = gamma * ZN[i] + beta  where gamma and beta stand for learnable parameters of model

如果 gamma = sqrt(sigma^2 + eplsion), beta = mean, 则 Ztilde[i] = Z[i], 所以我们用这两个参数可以调节下一层的均值和方差
 
batch-normal 通常和mini-batch一同运用，此时偏置项b没有意义，被beta取代


batch-normal原理：
【1】batch-normal 直观上将隐藏层的输出映射到指定范围内，有利于激活函数反向传播；
【2】更深层原因是，它可以使权重（比你的网络更滞后或者更深层）比如第十层的权重，更经受得住变化
"covariate shift": 如果已经学习到了 x->y的映射，如果x的分布改变，那么需要重新学习x->y
在深度学习中，某一层的输入的分布其实由于之前层的权重改变，正发生改变。所以会发生covariate shift问题。
所以batch-noarl就是把Z[l-1]层均值，std为固定，限制了前层的参数更新，减少了输入值的分布改变问题；所以之后的学习更加稳定；
【3】承担正则化作用：  
    -每一个mini-batch被sacled by mean/var computed on just that mini-batch
    -这添加了一些噪声给z[l]在那个minibatch上，所以和dropout相似，添加了一些噪声给每一个hidden layers;s activations
    -有轻微的正则化作用   
    
    
测试集和预测集的batch normal：需要单独估算mean 和 std，通常使用指数加权平均来估算，这个平均数涵盖了所有的mini-batch   
即：对mini-batch x{1}\x{2}\x{3}...
保存其mean{1}[l], std{1}[l] ...
之后使用指数加权平均，计算mean/std的这一层的平均值，作为该层的估计
"""

#%% softmax

"""
softmax生成多类别的相应的概率，其加和为1；

softmax： Z[l] = W[l] * A[l-1] + b[l]  (4 * 1) （设4为类别数）
activation function: t[l] = exp(Z[l]), 对Z的每一个元素进行指数化
A[l] = t[l]/sum(t[i])

set C = 4; y =[0,1,0,0] -> A[l] = [0.3,0.2,0.1,0.4]

loss(y_, y) = - sum(y[i]*log[yi_])
J(w,b,..) = 1/m * sum(loss(y_[i],y[i]))

反向传播的时候，dZ[l] = y_hat - y; 剩余和之前类似；

"""

