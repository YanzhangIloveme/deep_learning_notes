#%% 1 词汇表征-词嵌入
# 之前我们对词是用1-hot的表征；这种方法把每个词都孤立起来了，算法对相关的词无法准确的表示；
# 如 I want a glass of orange (juice) <-> I want a glass of apple ()
# 所以我们需要让orange和apple表征接近；

# 另一种词的特征表示方法：Featurized Representation
"""
        Man     Woman   King    Queen   Apple   Orange
Gender   -1       1    -0.95     0.97    0.00   0.01
Royal   0.01    0.02    0.93     0.95   -0.01   0.00
Age     0.03    0.02     0.7     0.69    0.03  -0.02
Food    0.09    0.01    0.02     0.01    0.95   0.97
...
"""

# 如果用这种表征方法，那么Apple和Orange就会比较接近；
# 那么之后的算法生成句子的时候比one-hot更加泛化，更加学习词的联系；
# 常用t-sne算法对词嵌入的结果可视化

#%% 2 使用词嵌入
# 使用人名实体判别的例子；
"""
Sally Jhonson is an orange farmer.
|       |      |  |   |       |
1       1      0  0   0       0

使用词嵌入的时候，那么对
Robert Lin is an apple farmer中，就更好的判断Robert Lin是人了；
通常词嵌入需要大量1B-100B的训练集，那么这样可以使用迁移学习，直接应用别人已经使用大量数据训练好的词表征情况

# 词嵌入迁移学习步骤:
## 1 Learn word embeddings from large text corpus(1-100B) or pre-trained embedding online
## 2 Transfer embedding to new task with smaller training set. 这个步骤可以将原来预训练的嵌入向量维度降低（如从10k维->300的向量表征）
## 3 Optional：Continue to finetune the word embeddings with new data.
(这个过程和人脸识别中人脸编码有异曲同工之妙，也可以从预训练模型进行迁移学习)
"""
#%% 3 Properties of word embeddings - 类比推理
# 类比推理很好理解，使用词嵌入，如果man->women; 则king->queen
# 加入有向量eman-ewomen = [-2, 0, 0, 0].T   eking-equeen = [-2,0,0,0].T
# 所以我们可以找一个算法，让emen-ewomen ~= eking-e? 即可；

"""
算法流程：
emen-ewomen ~= eking-e?
find word w： argmax(sim(ew, eking-eman+ewomen))
find the result: equeen
这种算法，准确率大概只有30~75%；最常用的相似度函数是Cosine-similarity
sim(u,v) = u.T * v / (||u||**2 * ||v||**2)
这种关系可以让模型学习 man->women = boy -> girl这种模式

"""

#%% 4 Embedding matrix - 嵌入矩阵
# 设字典有10000词
# 我们佳宁要学习一个300*10000的矩阵
"""
 ------------------------------------------
 ^  a   aaron  ...  orange  ... <unk>   ^
 |                                      |
300                                    300
 |                                      |
 V                                      V
 ------------------------------------------
 
如果orange为6527，则这个词的one-hot向量[0,0,0...1,0...,0]为1*10000的shape
所以matrix E * O -> (300*10000) * (10000,1) = (300,1),所得的就是该矩阵的orange这个的嵌入的vector
所以我们的目标就是学习一个这样的嵌入矩阵
"""

#%% 5 Learning word embeddings
# 从历史来看，整个词嵌入算法是一个由复杂逐步变简单的过程；我们从复杂算法开始，然后逐步研究新的简单而好用的算法。
# 如  I   want a  glass of   orage ___. 这个句子
#   4343 9665  1  3852  6163  6257
# 建立一个语言模型是学习词嵌入的好方法；
"""
我们来学习一个300维的向量，过程：
I       O4343 -> E -> e4343    |
want    O9665 -> E -> e9665    |
a       O1    -> E -> e1       | 训练一个网络模型[6*300] --> softmax() -> 预测___这个词，那么中间这个网络模型就是训练好的词向量
glass   O3852 -> E -> e3852    |
of      O6163 -> E -> e6163    |
orange  O6257 -> E -> e6257    |

那么有一个参数，如n-gram=4，则算法只看前4个词；中间的这个E是共享，通过反向传播可以更新嵌入的词；

【总结】
通过上下文设置目标词：如
context: last 4 words
context：4 words left & right（其中n-gram的4就是个超参数）
context: last 1 word
context: Nearby 1 word! (这个思想就是skip-gram)
"""

#%% 6 Word2Vec
# 引入：Skip-grams算法
# I want a glass of orange juice to go along with my cereal.
# 抽取上下文和目标词配对，构造成监督学习；
# 这里我们需要随机选个上下文，而不是固定的位置；根据词预测目标上下文
"""
基于Skip-grams的Word2vec模型训练过程：
从Oc -> E -> ec = E * Oc -> softmax -> y^
softmax: p(t|c) = e^thetat.T*ec / sum(e^thetaj.T*ec)
L(y^,y) = -sum(yi*logyi^)

但是这个算法有这么几个问题:
1 计算速度：每次都需要对词汇表全部的词汇总量求和，如果词汇表数量很大，那么计算量爆炸；（可以使用分层softmax的方法加速这个过程，或者更先进的方法：负采样）
2 如何对context c 上下文c进行采样？：避免太多的a/an/the 这种词
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
除了Skip-gram外，还有CBOW模型（Continuous Bag of Words ）：根据上下文预测目标词；

这种方式在2018年前比较主流，但是当 BERT、GPT2.0出来以后，已经不是效果最好的方法了
"""

#%% 7 Negative Sampling

# 构造一个监督学习，如orange juice -> 预测这个是否是一对上下文词（pair)；相反； orange king就是不相关的词；target 为0
# 正样本方法相同，负样本从字典里面随机选择；组成pairs；这个步骤重复K次；小数据集中k为5-20合适；大数据集k=2~5
"""
p(y=1|c,t) = sigmoid(thetat.T * ec)
ec->E->et -> [0,1,...]
所以不需要对全部10000个做softmax，只需要对k+1个做softmax即可；
负采样上，使用P(wi) = f(wi)^(3/4) / sum(f(wj)^3/4) 这个分布比较好
"""

#%% 8 Glove词向量 （global vectors for word representation)
# I want a glass of orange juice to go along with my cereal.
# 设 Xij = # times i appears in context of j
"""
流程：
1、根据语料库构建一个共现矩阵，矩阵中每一个元素Xij代表单词i和单词j在特定大小的上下文窗口内共同出现的次数。
（一般这个次数最小值是1，但是glove做了进一步处理，根据两个单词在上下文窗口的距离d，提出衰减函数decay=1/d用来计算权重。也就是距离越远的两个单词所占总计数权重越小）
2、构建词向量word vecort和共现矩阵之间的近似关系：Wi.T * Wj^2 + bi + bj^2 = log(Xij) 其中Wi和Wj^2是要求的词向量，bi和bj^2是两个词向量的偏置项
3、构造损失函数minimize sum_i.sum_j(f(x_ij) * (Wi.T * Wj^2 + bi + bj^2 - log(Xij))^2

这个损失函数是最简单的mean square loss，只不过在此基础上增加了一个权重函数f(X_{ij})，它的作用在于：对于在一个语料库中经常一起出现的单词（frequent occurrence），
    -这些单词的权重要大于那些很少在一起出现的单词（rare occurrence），所以这个函数是非递减函数；
    -这些单词的权重也不能太大（overweighted），当到达一定程度之后应该不再增加；
    -如果两个单词没有在一起出现，即X_{ij}=0，那么它们不应该参与到loss function 的计算中去，即f(0)=0
所以最后选择了分段函数： f(x) = (x/x_max)^a if x<x_max ; 1 otherwise

训练方法：虽然很多人声称Glove是一种无监督（unsupervised learning）的学习方式，即不需要人工标注数据，但实际上它还是有标签的，这个标签就是log(Xij)，
而向量和就是要不断更新学习的参数。因此，本质上它的训练方式和监督学习的训练方式没有什么不同，都是基于梯度下降的。

训练的具体做法是：采用AdaGrad的梯度下降算法，对矩阵中的所有非零元素进行随机采样，学习率设置为0.05，在vector size小于300的情况下迭代50次，其他大小的vector size迭代100次，直至收敛。

如果本身语料比较小，微调没什么作用，或者自己直接训练没有很强的算力，直接使用采用大数据进行预训练的glove词向量也会有比较好的效果。


"""

# Glove与LSA、word2vec比较

# LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量 表征工具，是基于co-occurence matrix的。其采用基于奇异值分解（SVD）
# 的矩阵分解技术对大矩阵进行降维，因为SVD的复杂度很高，所以计算代价比较大。此外，它对所有单词的统计权重都是一致的。
# word2vec最大的缺点是只利用了固定窗口内的语料，而没有充分利用所有的语料。
# 所以Glove是把两者的优点结合了起来。

#%% 9 情感分析 

# 一般情感分析需要较大的数据集，通过词嵌入方法，较小的数据集也可以在情感分析中获得不错的效果
# x -> y
# The dessert is excellent. -> 4
# Completely lacking in good taste. -> 1
# algo1: sum/avg方法：可对不同的vector sum或者average -> 接入softmax层->分类y；但是忽略了词序和词的关系，如not good，可能就认为只是good
# algo2: 使用rnn：vectors -> RNN -> 在最后一个输入后，输出y； 即nv1问题；

#%% 10 词嵌入消除偏见
# 如男人：程序员 -> 女人：家庭主妇；这种就是一种算法学习到的偏见。
# 设我们已经学习到了一套word vector
"""
1. Identify bias direction:可以将{e_he-e_she, e_male-e_female ...}收集一系列存在歧视的向量，求avg；所得到的趋势（或方向）就是嵌入向量的歧视方向
2. Neutralize（中和步）For every word that is not definitional, project to get rid of bias;将他们向歧视方向垂直的方向（非歧视方向）移动
3. Equalize pairs（均衡步）：使grandfather和grandmother向量和babysister这种词距离相似；

"""

