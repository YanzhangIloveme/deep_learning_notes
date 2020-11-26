# 计算机视觉引入：

"""
Image Classification
Object Detection
Neural Style Transfer
"""

#%% 1 边缘检测

# -> 使用卷积运算 based on a filter
# -> move filter by steps

# vertical edge detection

"""
10 10 10 0 0 0 
10 10 10 0 0 0                1 0 -1        0 30 30 0
10 10 10 0 0 0 --->  * --->   1 0 -1 --->   0 30 30 0
10 10 10 0 0 0                1 0 -1        0 30 30 0
10 10 10 0 0 0                              0 30 30 0
10 10 10 0 0 0
"""

# Horizontal edge detection

"""
10 10 10  0  0  0 
10 10 10  0  0  0                1 1 1         0   0   0   0
10 10 10  0  0  0 --->  * --->   0 0 0  --->  30  10 -10 -30
 0  0  0 10 10 10               -1-1-1        30  10 -10 -30
 0  0  0 10 10 10                              0   0   0   0
 0  0  0 10 10 10
"""

# Sobel filter

"""
1 0 -1
2 0 -2
1 0 -1
"""

# Scharr filter
"""
3  0 -3
10 0 -10
3  0 -3
"""

# CNN:使用bp算法更新的filter将更优于人工filter

#%% Padding操作

# 如果用 [n * n] filter by [f * f] -> 最后一共有 [(n-f+1) * (n-f+1)]种可能
# 这样两个缺点：

# 1) 不断进行边缘识别或者特征提取后，根据卷积操作，图片会不断变小
# 2) 落在图片边缘的像素被卷积操作的次数<<中间的像素，这样边缘的像素信息就被丢失了

# 操作：padding，如原来的6*6图片，在最外层包上一层像素 ，如1层(通常使用0填充，p=1，填充1层)，则变成8 * 8,
# 那么输出变成了 (n+2p-f+1) * (n+2p-f+1) -> (6 * 6) 保持原来的图片的尺寸了


# valid convolutions: No padding
# Same convolutions:  pading with (f-1)/2; output is same with original size
# f 通常是odd数： 1 维持卷积后对称 2 奇数可以有一个中心点，便于指出过滤器的位置；

#%% Strided Convolution 步长

"""
[n * n] filtered by [f * f] with padding p on strided = 2, Then you can get:
[(n + 2p -f)/2+1 * (n + 2p -f)/2+1 ]
如果无法整除，则向下取整；

数学定义中卷积对滤波器多进行了一个横向/纵向旋转操作：

 3 4 5        7 2 5
 1 0 2   - >  9 0 4
-1 9 7       -1 1 3 
其这样做法使得卷积操作： (A * B) * C = A * (B * C) 这样对一些信号处理可以更方便；

但是机器学习中我们通常不做这个旋转操作；所以有时也称为cross-correlation互相关
"""
#%% Convolutions over volumes (对高维卷积)

"""
        RGB                                    filter
    [6 * 6 * 3]             -->             [3 * 3 * 3]
height/width/channels                 height/width/channels

所以他们对channels必须相等；同时对其进行卷积；最后得出结果是 
[(n+2p-f)/strides + 1] * [(n+2p-f)/strides + 1] * nc   最后多个channel聚合了！ nc为filter的数量

所以如果只想过滤红色纵向；则三层filter可以设计为：

R [[1 0 -1]       G [[0,0,0]]*3   B [[0,0,0]]*3
   [1 0 -1]
   [1 0 -1]]


同时对多个过滤（不仅仅用一个filters时），可以将多个filter结果增加维度

如 [6 * 6 * 3] * [3 * 3 * 3] [纵向]  
               * [3 * 3 * 3] [横向向]
同时进行两个filter过滤对时候，将其结果进行叠加,最后得到 [3 * 3 * 2]
"""

#%% One layer of a convolution network

"""

如 [6 * 6 * 3] * [3 * 3 * 3]  + b1 (broadcast) -> Relu() -> [4 * 4]   |  --> [4 * 4 * 2]
               * [3 * 3 * 3]  + b2 (broadcast) -> Relu() ->  [4 * 4]  |

所以：
Z[1] = W[1] * A[0] + b[1]  -> 这里A[0]即是X [6 * 6 * 3], 这里W[1] 为 filter，和 A[0] 进行卷积操作
A[1] = g([Z[1]]) -> 激活

参数计算：如果10个filters 每个3*3*3+1（bias）, 则280个参数，不管输入的图片大小多大，其参数数量固定，这是卷积神经网络的一个特性->避免过拟合

编程惯例：
f[l] filter size
p[l] padding
s[l] stride
input: n_h[l-1] * n_w[l-1] * n_c[l-1]
output: input: n_h[l] * n_w[l] * n_c[l]  -> [(n+2p-f)/s[l] + 1]
Each filter: f[l] * f[l] * n_c[l-1]
activations: a[l] -> n_h[l] * n_w[l] * n_c[l]
A[l] -> m * n_h[l] * n_w[l] * n_c[l] 共m个变量
Weights: f[l] * f[l] * n_c[l-1] * n_c[l] (L层过滤器的数量)
Bias: [1,1,1,n_c[l]]

"""

#%% 简单卷积网络示意
# 如图像分类

# 39* 39 *3 -> n_H[0] * n_W[0] * n_C[0] = 39 * 39 * 3
# -> f[1] =3, s[1] = 1, p[1] = 0, n_C[1] = 10
# 39* 39 *3 -> n_H[1] * n_W[1] * n_C[1] = 37 * 37 * 10
# -> f[2] =5, s[2] = 2, p[2] = 0, n_C[2] = 20
# 39* 39 *3 -> n_H[2] * n_W[2] * n_C[2] = 17 * 17 * 20
# -> f[3] =5, s[2] = 2, p[2] = 0, n_C[2] = 40
# 39* 39 *3 -> n_H[3] * n_W[3] * n_C[3] = 7 * 7 * 40
# -> 平滑展开 7 * 7 * 40 = 1960
# -> softmax()

# 一个典型的cnn网络有这么几层：
# 1 Convolution
# 2 Pooling
# 3 Fully Connected

#%% 池化层

# reduce size of representation
# speed up computation
# increase features robust

# max pooling

"""
1 3 2 1                          
2 9 1 1   ----MAX Pooling ---->  9  2 
1 3 2 3      f = 2 / s = 2       6  3
5 6 1 2 

Max pooling 直观理解： 只要任意象限提取到了某个特征，他就被保留在最大池化输出内
如果在过滤器中提取到某个特征，那么保留其最大值；
max pooling 过程没有任何参数需要学习
如 原始为5 * 5 | (f = 3) & (s = 1)  -> [(5-3+1) * (5-3+1)]
其公式和卷积操作相同；

同理，如果原始数据多个通道，则max pooling相应的需要同样数量的通道；
一般max pooling 不用padding，除了一种例外；后面讲到；
"""


# average pooling
# 原理和max pooling类似，但是通常max pooling更常用，但是如果有时候深度很深的网络，可以用平均池化
# 分解规模为 7 * 7 * 1000的网络表示层 -> 1 * 1 * 1000

#%% CNN网络示意

"""
由Lenet-5 简化：

Input           (32, 32, 3)         |    0
CONV1(f=5, s=1) (28, 28, 8)         |    208 = 5 * 5 * 8 + 8 
POOL1(f=2, s=2) (14, 14, 8)         |    0
CONV2(f=5, s=1) (10, 10, 16)        |    408 = 5 * 5 * 16 + 8
POOL2(f=2, s=2) ( 5,  5, 16)        |    0
FC3             (120, 1)            |  48001 = 120 * 400 + 1
FC4             (84,  1)            |  10081 = 84 * 120 + 1
Softmax         (10,  1)            |  841


# 可以看到大部分的参数在全连接层，通过卷积操作大大减少了参数的数量
"""

#%% CNN优势

# 1 参数共享 -> 训练出来的filter在图片某个部分有效的同时可能对其他部分也有效果
# 2 稀疏连接 -> 卷积操作中的只与filter相关的几个数相关，其余的并不参与运算；




