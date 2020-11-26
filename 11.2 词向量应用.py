# 1 - 词向量运算
# 因为词嵌入的训练是非常耗资源的，所以大部分人都是选择加载训练好的词嵌入数据。在本博客中，我们将学习到：
"""
如何加载训练好了的词向量
使用余弦相似性计算相似度
使用词嵌入来解决“男人与女人相比就像国王与____ 相比”之类的词语类比问题
修改词嵌入以减少性别偏见等
"""
import numpy as np
import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets')
from wordvector import w2v_utils

# 接下来就是加载词向量了，这里我们使用50维的向量来表示单词：
words, word_to_vec_map = w2v_utils.read_glove_vecs('datasets/wordvector/data/glove.6B.50d.txt')

# words：单词的集合
# word_to_vec_map ： 字典类型，单词到GloVe向量的映射
# 因为独热向量不能很好地表示词语词之间的相似性，所以使用了GloVe向量，它保存了每个单词更多、更有用的信息，我们现在可以看看如何比较两个词的相似性。

#%% 1.1 余弦相似度
def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度

    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量

    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """
    distance = 0

    # 计算u与v的内积
    dot = np.dot(u, v)

    # 计算u的L2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))

    # 计算v的L2范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    # 根据公式1计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)

    return cosine_similarity

# test
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]
fuck = word_to_vec_map["fuck"]
suck = word_to_vec_map["suck"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
print("cosine_similarity(fuck, suck) = ",cosine_similarity(fuck, suck))
print("cosine_similarity(fuck, mother) = ",cosine_similarity(fuck, mother))

#%% 1.2 - 词类类比
# 在这里，我们将学习解决“A与B相比就类似于C与____相比一样”之类的问题，打个比方，“男人与女人相比就像国王与 女皇 相比”。实际上我们需要找到一个词d
# 然后 e_b - e_a + e_c ~= e_d
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题

    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """

    # 把单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    # 获取全部的单词
    words = word_to_vec_map.keys()

    # 将max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None

    # 遍历整个数据集
    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word

# test

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))

#%% 1.3 - 去除词向量中的偏见（选学-思路）
"""
在这一部分，我们将研究反映在词嵌入中的性别偏差，并试着去去除这一些偏差，除了学习这个内容外，这一节还可以磨炼你对单词向量的直觉，这部分包含有线性代数，
不是很难，如果你没有学习过线性代数，那么你可以跳过这一节，你也可以继续深入下去。

我们首先来看一下包含在词嵌入中的性别偏差，我们首先计算一下 g = e_women - e_man, 其中e_women是单词woman的对应词向量；
这样的结果g粗略的包含了性别这一概念。如果计算g1 = e_mother - efather 与 g2 = e_gial - e_boy的平均值，可能更准确些；
"""
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)
# 现在，我们考虑不同单词与g的余弦相似度，考虑相似度的正值与余弦相似度的负值之间的关系。
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))

# 正如我们所看见的，女性的名字与g gg的余弦相似度为正，而男性为负，这也不出乎人的意料，我们来看看其他词

word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))


#   发现了吗？比如“computer”就接近于“man”，“literature ”接近于“woman”，但是这些都是不对的一些观念，那么我们该如何减少这些偏差呢？
#   对于一些特殊的词汇而言，比如“男演员（actor）”与“女演员（actress）”或者“祖母（grandmother）”与“祖父（grandfather）”之间应该是具有性别差异的，
#   但是其他的词汇比如“接待员（receptionist）”与“技术（technology ）”是不应该有性别差异的，当我们处理这些词汇的时候应该区别对待。


#%% 1.4 消除与性别无关的词汇的偏差
# 如果我们使用的是50维的词嵌入，那么50维的空间可以分为两个部分： 1 偏置方向 2 剩下49维方向（这49维方向和偏置方向正交）
# 现在我们要实现**neutralize()** 函数来消除包含在词汇向量里面的性别偏差，给定一个输入：词嵌入（embedding）
# 那么我们可以使用下面的公式计算：
# e_bias_component = np.dot(e,g)/l2_norm(g) * g
# e_debiased = e - e_bias_component
# 其中，e_bias_component是e在g方向的投影；

def neutralize(word, g, word_to_vec_map):
    """
    通过将“word”投影到与偏置轴正交的空间上，消除了“word”的偏差。
    该函数确保“word”在性别的子空间中的值为0

    参数：
        word -- 待消除偏差的字符串
        g -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_debiased -- 消除了偏差的向量。
    """
    e = word_to_vec_map[word]
    # 根据公式2计算e_biascomponent
    e_bias_component = np.divide(np.dot(e,g), np.square(np.linalg.norm(g))) * g
    # 根据公式3计算e_debiased
    e_debiased = e - e_bias_component
    return e_debiased

e = "receptionist"
print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word_to_vec_map["receptionist"], g)))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))

#%% 1.5 性别词的均衡算法
"""
接下来我们来看看在关于有特定性别词组中，如何将它们进行均衡，比如“男演员”与“女演员”中，与“保姆”一词更接近的是“女演员”，我们可以消去“保姆”的性别偏差，
但是这并不能保证“保姆”一词与“男演员”与“女演员”之间的距离相等，我们要学的均衡算法将解决这个问题。
均衡算法背后的关键思想是确保一对特定的单词在歧视方向的g上距离保持相等
线性代数关键方程如下：
u = (e_w1 + e_w2)/2
u_b = np.dot(u , bias_axis)/l2_norm(bias_axis) * bias_axis (两者平均值的投影)
u_t = u - u_b
e_w1b = np.dot(e_w1,bias_axis)/l2_norm(bias_axis) * bias_axis (1的投影)
e_w2b = np.dot(e_w2,bias_axis)/l2_norm(bias_axis) * bias_axis (2的投影)
e_w1b_corr = sqrt(abs(1-l2_norm(u_t))) * (e_w1b - u_b)/|(e_w1 - u_t)-u_b|
e_w2b_corr = sqrt(abs(1-l2_norm(u_t))) * (e_w2b - u_b)/|(e_w1 - u_t)-u_b|
e1 = e_w1b_corr + u_t
e2 = e_w2b_corr + u_t
"""


def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。

    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor")
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # 第1步：获取词向量
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    # 第2步：计算w1与w2的均值
    mu = (e_w1 + e_w2) / 2.0
    # 第3步：计算mu在偏置轴与正交轴上的投影
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_orth = mu - mu_B
    # 第4步：使用公式7、8计算e_w1B 与 e_w2B
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    # 第5步：根据公式9、10调整e_w1B 与 e_w2B的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1-np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1B-mu_B, np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1-np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2B-mu_B, np.abs(e_w2 - mu_orth - mu_B))
    # 第6步：根据公式9、10调整e_w1B 与 e_w2B的偏置部分
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    return e1, e2
# test
print("==========均衡校正前==========")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("\n==========均衡校正后==========")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))