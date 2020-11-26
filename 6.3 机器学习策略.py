# 1 正交化 Orthogonalization

"""
Orthogonalization is a system design property that assures that modifying an instruction or a component of an algorithm
will not create or propagate side effects to other components of the system.

This becomes easier to verify the algorithm independently from one another, it reduces testing and development time.


fit training set well on cost func: [bigger network/adam]

fit dev set well on cost func:[regularization/bigger training set]

fit test set well on cost function: [bigger dev set]

performs well in real world: [change dev set/change cost function]

"""

#%% 2 single number evalueation metric
# precision/recall
# f1-score

#%% 3 训练/开发/测试

# choose a dev set and test set to reflect data you expect to get in the future and consider important
# to do well on

#%% 4 当数据集中分类的权重不一致的时候，可以考虑改变损失函数

# w[i] = (1 if x[i] is porn; 10 if x[i] is porn)
# J = sum(w[i] * Loss(y_hat, y))
# metric必须和实际业务吻合

#%% 5 Bayes optimal error 贝叶斯最优误差

"""
机器学习超过人类水平的时候，其提升进展非常缓慢；趋向贝叶斯最优误差。
如果没有超过，有些手段可以继续提升；

1. get labeled data from humans
2. gain insight from manual error analysis: "为什么人类分的正确？"
3. better analysis of bias/variance
"""

#%% 6 可避免误差

"""
Humans level: 1%
training error: 8%
dev error: 10%
reduce bias
=================
Humans level: 7.5%
training error: 8%
dev error: 10%
reduce variance
now this 0.5% diff refers to avoidable error
"""

#%% 7 进行误差分析
# 设计误差分析矩阵，将错误分类的案例依次统计是什么类型的错误
# eg:
"""
1    dog    blur   fliter  comment
2    True
3                  True
4
5
6

sum  ==     ==       ==   

===========================
"""

#%% 8 错误标记的数据
"""
DL algos are quite robust to random error in the training sets;
if these error-labelled data occur in testing sets, find it in error analysis

但是需要注意：

1. 对dev/test集合使用相同的操作，确保他们处于同一个dist之中
2. 不仅仅要注意算法在错误类别的分类情况，也要注意算法在正确类别的分类；
3. 训练集、测试集数据可能来自不同的分布之中
"""

#%% 9 在不同的划分上进行训练并测试

"""
case1: 如果有20w只猫照片web爬取，1w只是app照片（最终落地的版本），我们的目标是针对app照片；
那么 opt1: 21w张混合，然后分train/dev/test   --> 不好，因为最终test/dev集的分布和最后落地的分布非常不同
opt2: 20.5w train, 0.25/0.25 from app照片 --> 更合适


Case2: Speech recognition;
train: Purchased data/Smart spearker control/Voice keyboard
dev/test: Speech activated/rearview mirror

"""


#%% 10 不匹配数据划分偏差和方差

"""
假设算法在training error 1%; 在dev error 9%; 可能过拟合，也可能dev的分布发生了变化
这个时候，我们也可以考虑多划分一个training dev set; 这个set和train必须同分布，但是没有参加训练，
训练好之后，在这个set上进行预测，看看是var问题还是分布问题；
"""

#%% 11 data mismatch 问题

# 1 尝试理解训练集和dev/test集的差异性
# 2 使训练数据和dev更相近，或者收集更多和dev/test相近的数据(如模拟噪音数据)

# 如：clean audio + car noise = Synthesized in-car audio

#%% 12 迁移学习

# 其他领域的深度学习经验，可以复用在另外的领域上，这统称为迁移学习
# 如训练好的cat/dog classifer，加以修改再训练，可以使用在x光检测上；
# 我们需要把一个训练好的模型拿到，然后把最后一层的权重初始化为随机数，然后用新的数据集（x,y)来训练，这就是迁移学习
# 如果新的数据集不多，那么只训练最后一层/或者最后两层的参数；
# 如果数据很多，那么有资本训练前面所有的层（这种方法有时候叫预训练）；之后用新数据集更新参数有时候叫fine-tune 微调

# 原理：有很多低层次特征，如边缘检测、曲线检测、阳性对象检测等，有助于新一轮的学习；
# 迁移学习在这种情况有效：
# 1. TASK_A 和 TASK_B有相同的输入input x
# 2. TASK_A 数据量>>TASK_B 数据量
# 3. Low-level 的features from TASK_A is helpful for TASK_B


#%% 13 多任务学习

# 如无人驾驶，需要同时检测 pedestrians, cars, stop signs, traffic lights等等；
# 如： Y = [0, 1, 1, 0]
# 则，此时损失函数为：J(w,x ...) = 1/m * sum(i=1..m) * sum(j=1..m) * L(y[j][i], y[j][i]_hat)
# 多任务学习在以下三点满足时才有效：
# 1 training on a set of tasks that could benefit from having shared lower-lvel features;
# 2 Usually: Amount of data you have for each task is quite similar
# 3 can train a big enough neural network to do well on all the tasks

#%% 14 端对端学习 end-to-end deep learning

# 一些系统中需要满足数据多个阶段的处理，端对端学习就是忽略中间的数据处理阶段，直接用一个大的深度学习模型来代替，称之为端对端学习；

# case1: speech recognition
# 传统方法： Audio -> MFCC算法 -> features -> ML algos -> phonemes(音位) -> words -> transcript
# e2e-学习: Audio ---------------------------------------------------------------> Transcript
# 其挑战在于：需要大量的数据；

# case2: 人脸识别系统
# 由于人脸识别存在多个角度，前后位置不同等情况，直接端对端学习存在较大难度；通常架构设计为：
# 应用人脸提取软件/算法，检测人类，检测到了人脸，则放大照片，使其居中显示，然后再放到神经网络之中；
# （步骤2中，通常是输入两张图，神经网络来判断这两个图是否是同一个人）

# case3: 机器翻译
# 目前端对端学习已经很好达成了；因为目前(x,y)翻译对儿数据量很大

# case4: 手骨骼判断儿童年龄
# 目前端对端学习效果不好，因为没有足够的数据；
# 通常，我们从x光中--> 切分骨骼--> 测量该骨头的长度 --> 比对经验这个长度通常是多少岁的年龄

#%% 15 end-to-end 方法pros and cons

"""
Pros: 
    - let the data speak
    - less hand-designing of components needed
Conds:
    - may need large amount of data
    - excludes potentially useful hand-designed components
keys:
    - Do you have sufficient data to learn a func of the complexity needed to map x to y
    
"""


