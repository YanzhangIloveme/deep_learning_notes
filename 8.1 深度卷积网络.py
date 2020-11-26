#%% Guildline

"""
LeNet-5
AlexNet
VGG
ResNet（152）
Inception
"""

#%% 1 LeNet -5

"""
input(32*32*1) -> conv(5*5*1*6, s=1) -> avg pool (f=2 s=2) -> conv(5*5*1*16, s=1) -> avg pool (f=2 s=2)
-> FC(120) ->fc(84) -softmax-> yhat

在当时，激活函数还用的是sigmoid/tanh 而不是Relu
"""

#%% 2 AlexNet
"""
input(227*227*3) -> conv(11*11*3*96, s=4) -> max pool(3*3*96, s=2) -> conv(5*5*256 same) -> max pool (3*3*256 s=2)
-> conv(3*3*384 same) -> conv(3*3*384 same) -> conv(3*3*256 same) -> max pool(3*3*256 s=2) -> FC[9216] -> FC[4096]
-> FC[4096] -> softmax(1000) 

local response normalization : 对某个层多个如13*13*256， 在256方向归一化（后来研究者认为LRN起不到太大作用）
"""

#%% 3 VGG-16

# VGG只用小的conv 3*3 filter s=1 same -> max-pool = 2*2, s=2   --> 多个小的conv filter = 大对conv filter 但是大大降低参数数量
"""
input(224*224*3) -> conv(3*3*3*64, s=1)  * 2次 -> pool (之后shape为112 * 112 * 64) ->  conv(3*3*3*128, s=1) *2 -> pool (56*56*128)
->  conv(3*3*3*256, s=1) * 3 -> pool(28*28*256) ->  conv(3*3*3*512, s=1) * 3 -> pool (14*14*512) ->  conv(3*3*3*512, s=1)
-> pool(7*7*512) -> FC(4096) -> FC(4096) -> Softmax(1000)
"""

#%% 4 ResNet
# -> skip connections (跳远连接)： 可以从某一个网络层获取激活，迅速反馈给另外一层，基于此可以训练更深对网络
# -> a[l] -> a[l+1] -> a[l+2]
# -> 传统对连接方式：a[l] -> Linear -> Relu -> a[l+1] -> Linear -> Relu -> a[l+2]
# -> 残缺块连接方式：a[l] -> Linear -> Relu -> a[l+1] -> Linear -> Relu -> a[l+2]
#                   ｜________________________________________｜ 直接连接到l+2， 在Relu到非线性激活前加上a[l]
# a[l+2] = g(Z[l+2]+a[l])
# 使用这样的skip connections可以训练更加深层到神经网络

# Why ResNet work?

"""
a[l+2] = g(Z[l+2] + a[l]) = g(W[l+2]*A[l+1]+b[l+2] + a[l])
如果对W进行权重缩减，则W[l+2]则会压缩，如果W[l+2]=0, b[l+2]=0, 则a[l+2] = a[l]
此外，需要保证a[l+2]和a[l]维度一致，一种做法是same卷积操作，另一种操作是在a[l]前加一个(a[l+2]层，a[l]层)的矩阵Ws
即 a[l+2] = g(Z[l+2] + a[l]) = g(W[l+2]*A[l+1]+b[l+2] + Ws* a[l])，此时不需要对Ws进行操作；

"""


#%% 5 网络中对网络以及1*1卷积--> network in network

"""
1.改变Nc： 如原始为28*28*192, 我们可以使用1*1*32filters 将Nc压缩至32；
2.增加非线性：即使1*1 filters的Nc相同，也对原始对输入进行了一个非线性的函数变换；
3.跨通道信息交互(channal变换)，使用1x1卷积核，实现降维和升维的操作其实就是channel间信息的线性组合变化
"""

#%% 6 inception网络/GoogleNet

"""
如又一个28*28*192的输入，inception网络/inception层是代替人工来确定卷积层或者是否需要创建pooling
如果使用1*1*64 filters -same-> 28*28*64
如果使用3*3*128 filters -same-> 28*28*128
如果使用5*5*32 filters -same-> 28*28*32
如果用pooling: -same-> 28*28*32
然后在Nc上: 将上面的多个输出在Nc上叠加；64+128+32+32=256 -> 最后输出是28*28*256， 这一层即是Inception层

不需要人工设计网络层，而是让机器自己学习；但是问题也很明显，计算成本很高；
如执行5*5*32卷积； 每个filters的size为28*28*192
则乘法计算为28*28*32 * 5*5*192 = 120M 乘法运算；（使用1*1卷积可以大大降低这个计算成本）


使用1*1*16 filters将28*28*192 -> 28*28*16再使用5*5*32卷积；
则计算成本为：28*28*32 * 5*5*16 + 28*28*16 * 192*1*1 = 2.4M + 10.0M = 12.4M， 比120M小了近10倍；
"""

"""
构建Inception网络：

previous activation - > 1*1 -> 3*3         |
                    - > 1*1 -> 5*5         | Channel
                    - - - - -> maxpool     | Concat
                    - - - - -> 1*1         | 

则inception network是多个inception block 的堆积；
其中有些分支，这个分支是直接连接softmax来预测结果；从而防止过拟合；

"""

#%% 7 迁移学习

"""
如要进行猫咪分类：tigger/misty/neither 三分类；但是tigger/misty照片不多；

则从网上下载一些开源的模型即其权重；如1000分类；

则可以去掉最后的softmax层，换成自己的；然后可以把中间的层全部固定参数（freeze their weights）
可以设置 trainableParameter = 0 这种参数来完成；

另一种技巧是，把fixed层的输出保存在disk上，fiexed的部分其实就是一个函数，保存他的结果，再直接训练；
如果训练集很多，那么可以freeze前面基层较少的层，然后后几层重新训练；
"""

#%% 8 数据扩充

"""
--> 变换
1 mirroring
2 random cropping
3 Rotation
4 Shearing
5 local warping

---------------
--> 色彩转换
1 在各个通道上 +/- 20；
2 PCA color augmentation

"""

#%% 9 一些技巧

"""
1. ensembling：train several networks independently and average their outputs
2. Multi-crop at test time: run classifier on multiple versions of test images and average results (10-crop)
3. use architectures of networkds published in tthe literature
4. use open source implementations 
5. use pretrained models and fine-tune on your dataset


"""