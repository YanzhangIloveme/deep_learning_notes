#%% 1 多层神经网络的表达
'''
在之前的网络中，我们用了一层网络完成了逻辑回归，即 【X、b、w】-> z = w.T * X + b -> a = sigmoid(z) -> L(a, y)

如果增加一层，那么架构为：

2-layer nn

x1 -> a[1,1]
   \         \
x2 -> a[1,2]  -> a[2,1] -> L(a, y)
   /         /
x3 -> a[1.3]

input | hidden| output
layer | layer | layer
a[0]    a[1]     a[2]

'''

#%% 2 多层神经网络的向量化

# X (3,100) ; x1, x2, x3 3个特征，100个sample
# w1 (3,4) ; 4个神经元，分别提取3个特征
# b1 (4,1) *100; 4个神经元，广播到100个sample

# output a1: sigmoid(w1.T * X + b1 = (4,3)*(3,100)+(4,100) = (4,100))
# -----------------
# w2 (4,1); 四个神经元，转化为1个神经元
# b2 (1,1)；一个神经元，广播到100个sample
# output a2: sigmoid(w2.T * a1 +b2 = (1,4) * (4,100) + (1,100) = (1,100))

#%% 3 激活函数
# 3.1 sigmoid func: a=1/(1+exp(-z))
"""
其求导结果dg(z)/dz = exp(-x)/(1+exp(-x))^2
又因为：1-g(z) = exp(-x)/(1+exp(-x)), 所以g(z)*(1-g(z)) = dg(z)/dz
即 y*(1-y)
"""

# 3.2 tanh func: (always better than sigmoid func): a=[exp(z)-exp(-a)]/[exp(z)+exp(-a)]
"""
其函数为sigmoid的向下平移，再缩放到-1，1； 均值为0；有类似数据中心化的作用
其导数为：dg(z)/dz = 4/[exp(z)+exp(-z)]^2
又因为 1-y = 2*exp(-z)/[exp(z)+exp(-z)]; 1+y = 2*exp(z)/[exp(z)+exp(-z)]
即 (1-y)*(1+y) = dg(z)/dz -> 1-tanh^2
有一个例外，当输出层希望在（0，1）概率分布之间当时候，使用sigmiod激活函数
"""

# 3.3 rectified linear unit Relu（修正sigmoid和tanh在太大和太小梯度消失当情况）:a= max(0,a)
# 当z=0的时候，导数不存在；但编程时候，可以约定，当z非常小当时候，（如1e-7)，令其导数为0
"""
dg(z) = 1 when z>0 else 0
其中一个缺点是当z小于0的时候，导数为0；但是一般实践中，有足够多神经元，使z大于0；为了应对这个引入了leaky Relu
当z小于0的时候，使其导数稍微大于0；
由于z大于0的时候导数为1>>0,其训练速度远远大于sigmoid和tanh

"""
# 3.4 Leaky Relu: a=max(0.0..1*z, z)
"""
导数求导和容易；
"""

#%% 4 为何nn必须需要非线性激活函数
# 否则其线性关系的组合，最后仍然是线性模型；隐藏层没有一点用；

#%% 5 Gradient Descent for nn
# parameters: w[1], b[1], w[2], b[2]
# n_x/n[0] = nums of features
# n_1/n[1] = nums of hidden units
# n_2/n[2] = nums of output units
# cost func J(params) = 1/m* sum(L(y^,y)

"""
计算梯度的公式
Z[1] = w[1]*X+b[1]    
A[1] = g[1](Z[1])
Z[2] = w[2]*A[1]+b[2]
A[2] = g[2](Z[2])
------------------------
Y：（1,m)矩阵
dZ[2] = A[2]-Y (sigmiod）
dw[2] = 1/m * dZ[2] * A[1].T
db[2] = 1.m * np.sum(dZ[2], axis=1, keepdims=True) axis=1-->水平方向求和；keepdims使结果不是(n,)，而是（n[2],1)
dZ[1] = w[2].T * dZ[2] * dg[1](z[1]) -- (n[1],m) 和 (n[1],m)的逐个元素乘积
dw[1] = 1/m * dZ[1] * X.T
db[1] = 1/m * np.sum(dz[1], axis=1, keepdims=True)
"""

#%% 6 初始化参数

# 初始化参数w不可以全部为0，否则所有hidden units都在计算同一个值
# 通常w[1] = np.random.randn((2,2)) * 0.01 初始化为一个小的数，因为这时候梯度较大
# b[1]=np.zeros((2,1))