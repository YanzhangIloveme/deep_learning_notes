# 1 Examples of sequence data

# (1) Speech recognition -> 声波曲线 -> "The quick brown fox jumped over the lazy dog"
# (2) Music generation -> music
# (3) Sentiment classification -> "there is nothing to like in this movie" -> negative
# (4) DNA sequence analysis -> AGCCCCTGTGAGGAACTAG -> AG[CCCCTGTGAGGAACT]AG
# (5) Machine Translation -> Voulez-vous chanter avercmoi -> Do you want to sing with me?
# (6) Video activity recognition -> vedio -> running
# (7) Name entity recognition -> Yesterday, Harry Potter met Hermione Granger -> Yesterday, [Harry Potter] met [Hermione Granger]

#%% 2 数学符号
# 命名实体识别问题；自动识别句中实体名称，常常用于搜索引擎，如识别过去24h内所有新闻提到的人名；
# eg x: Harry Potter and Hermione Granger invented a new spell
#    x:[x1, x2 ,x3...x9], Tx=9，共9个元素
#    y: [1 1 0 1 1 0 0 0 0] Ty=9
# 可以用来查找人、公司、国家、货币等；

"""
# 如何表示句子中的词：
1 构建词表Vacabulary = [a,Aaron,...Harry...] 其位置分别为[1,2,...]，对商用来说，3w-5w词汇即可；
2 使用one-hot表示词，如x1是Harry, 它是[0,0,...,1,0,0..0],其余亦然
3 如果遇到不在词表的单词，使用标记unknown表示

"""

#%% 3 循环神经网络模型

# 1 why not standard networks?
# (1) inputs, outputs can be diff lengths in different examples;
# (2) Doesn't share features learned across different positions of text; (如x1学到的harry在其他位置可否也被标记为人名一部分)

# 2 循环神经网络结构：
# (1) 每一层除了输入x[t]外，还输入上一个x[t-1]的激活值，输出y[t]，
# (2) 第一个输入我们需要编造一个激活值x[0]，通常是0向量
# (3) 参数空间：Wax -> input的参数；Waa -> 上一个时间步输入的参数
# (4) 使用BRNNs来处理双向问题；
# ————————————————————————————————————————————————————————————————————
"""
a<t> = g(Waa * a<t-1> + Wax * x<t> + ba) 通常应用tanh，也有用relu的
y<t> = g(Wya * a<t> + by>
通常我们把[Waa | Wax] 放置一起；用Wa表示；同理，[a<t-1>, x<t>] 放置一起处理， 则[Waa | Wax] * [a<t-1>, x<t>] = Waa * a<t-1> + Wax * x<t> 

"""

#%% 4 RNN: Backpropagation through time
# 反向传播的方向和向前传播正好相反；
"""
设有x<1>,...x<t>.
x<1> -> a<1> -> L<1>...
x<n> -> a<n> -> L<n>
Loss = L<1> + ... L<n>
"""

#%% 5 不同类型的RNN结构，参考《循环神经网络不合理的有效性》
# [1] 实体提取：many-to-many结构；Tx=Ty
"""
Input:  x1, x2...xt
Output: y1, y2...yt
"""

# [2] Sentiment classification：many-to-one结构
"""
X = text
y = 0/1
x<1> -> x<2> -> ... -> x<t>
                        |
                       y<t>
只在句末输出，这样输入就是整个句子；
"""

# 【3】one-to-one结构：没啥可讨论的，基础的NN
# 【4】Music generation: One-to-Many结构
"""
X = null/one words
y = 0/1
x<1> 
 |
y<1> -> y<2> -> ... -> y<t>
其中有个技术是，通常把y<t-1>也当做输入，喂给下一层t
"""

# 【5】机器翻译，x，y不对等；Many-to-Many
"""
X = text
y = 0/1

x<1> -> x<2> ->| ... -> x<t> ----decoder----
               |         |                  |
____encoder____|        y<1> -> y<2> -> y<n>

"""

#%% 6 语言模型和序列生成
# Speech recognition
"""
在语音识别中，某句语音有两种翻译：
P(The apple and pair salad) = 3.2 * 10^-13
P(The apple and pear salad) = 5.7 * 10^-10
选择概率最大的语句作为正确的翻译
即，语言模型是P(sentence) = ? 
给定一个序列y<1>...y<t>,语言模型估计各个单词出现的可能性；

语言模型训练方法：
Training sets: large corpus of english text;
如：Cats average 15 hours of sleep a day，构建词典，将每个词构建为一个one-hot向量
然后，需要定义句子的结尾，通常增加一个额外的标记叫EOS；
y<1> -> cats
y<2> -> p(average|cats)
...
y<t> -> P(EOS|...)
"""

#%% 7 新序列采样
# 序列模型模拟了任意特定单词序列的概率，要做的是对这些概率分布进行采样来生成一个新的单词序列。
# 第一步要做的是对想要模型生成的第一个词进行采样，输入x<1>=0, a<0>=0,对第一个时间步得到的所有可能的输出，是经过softmax层出的概率，然后
# 根据这个概率分布进行随机采样。对这个向量使用np.random.choice，来根据向量中这些概率分布采样；

# 对之后的时间步，y<1>作为输入，softmax同样对y<2>进行采样；

# c除了基于字典；也可以对字符进行采样，即只有26个英文字母；
# 优点：不会出现未知的标志；基于字符的语言会将mau这样的序列实为非零序列，基于词汇语言由于mau不在词典，会标记unk
# 缺点：最后会得到太多太长序列，基于字符的语言模型在捕捉句子的依赖关系上不如基于词汇；且计算成本较高；

#%% 8 RNN的梯度消失和梯度爆炸；
# 编号1cat是单数，应该用was，编号2 cats是复数，用were
# 这个例子中的句子有长期的依赖，最前面的单词对句子后面的单词有影响。但基本的RNN模型（编号3）不擅长捕获长期依赖效应
# RNN反向传播很困难，会有梯度消失的问题，后面层的输出误差（编号6）很难影响前面层（编号7）的计算。
# 即很难让一个神经网络能够意识到它要记住看到的是单数名词还是复数名词，然后在序列后面生成依赖单复数形式的was或者were。

# 在反向传播的时候，随着层数的增多，梯度不仅可能指数型的下降，也可能指数型的上升。梯度消失在训练RNN时是首要的问题，不过梯度爆炸也会出现，
# 但是梯度爆炸很明显，因为指数级大的梯度会让参数变得极其大，以至于网络参数崩溃。参数大到崩溃会看到很多NaN，或者不是数字的情况，这意味着网络计算出现了数值溢出
# 解决方法：用梯度修剪。梯度向量如果大于某个阈值，缩放梯度向量，保证不会太大
# 但是梯度修剪无法解决梯度消失问题，所以接下来引入GRU单元；

#%% 9 GRU单元（Gated Recurrent Unit）
# 门控循环单元：改变了RNN的隐藏层，使其可以更好地捕捉深层连接，并改善了梯度消失问题
# 在经典RNN网中，a<t> = g(Wa[a<t-1>, x<t>] + ba)
# GRU引入一个新变量c：叫做memory cell，如主语单数，这里需要记忆这个c<t>，c<t>记录单复数a<t>;
# c<t>~ = tanh(Wc[c<t-1>, x<t>] + bc)
# 下表为u：update门；u = sigmoid(Wu[c<t-1>,x<t>)+bu)
# 那么GRU核心就是用c<t>~更新c<t>，然后门u的作用是判断是否更新它；
"""
GRU模型综述：
添加记忆细胞C和门gammaR，表达式如下：
C~<t> = tanh(Wc * [gammaR * C<t-1>, x<t>] + bc) where C<t-1> = a<t-1> （包含上一层C<t-1>和这层输入x<t>，还加上了一个新门：gammaR作为记忆）
gammaU = sigmoid(Wu[C<t-1>,x<t>]+bu) (使用上一层记忆cell和本层xt，加工为update门，决定是否更新)
gammaR = sigmoid(Wr[C<t-1>,x<t>]+br) (使用上一层记忆cell和本层xt，加工计算出下一个C~<t>和C<t-1>有多大相关性)
C<t> = gammaU * C~<t> + (1-gammaU)*C<t-1>
a<t> =C<t>；

例如：The cat, which already ate ..., was full.
      C<t> = 1
C<t>如cat是单数，那么记忆cell会一直记住，直到后面was。到was的时候，如果门决定要更新（gammaU接近1）那么C<t> 更偏向于C~<t>而非C<t-1>;
而cat和was中间的部分，gammaU应接近0，使其保证C<t-1>信息不丢失；这种情况下，由于gammaU接近0，1-gammauU接近1，那么C<t-1>的梯度得以很好的保留
"""

#%% 10 LSTM（比GRU更加有效）
"""
C~<t> = tanh(Wc * [a<t-1>, x<t>] + bc) 和GRU比，输入不是处理后的记忆细胞，而是直接是上一个激活层
gammaU = sigmoid(Wu[a<t-1>,x<t>]+bu) 和GRU比，输入不是处理后的记忆细胞，而是直接是上一个激活层 【更新门】
# LSTM中不仅仅是一个门控制是否更新了！GRU版本中C<t> = gammaU (记忆门)* C~<t> + (1-gammaU) （遗忘门）*C<t-1>
# 在LSTM中遗忘门不再是1-gammaU，而是新定义的,并且多加了一个输出层！
gammaF = sigmoid(Wu[a<t-1>,x<t>]+bf) 【遗忘门】
gammaO = sigmoid(Wo[a<t-1>,x<t>]+bo) 【输出门】
C<t> = gammaU * C~<t> + gammaF * C<t-1>
a<t> = gammaO * tanh(C<t>)

一些LSTM的变体是引入了peephole，门的设置不仅仅依赖a<t-1>和x<t>，还有c<t-1>，这个叫peephole connection
gammaX = sigmoid(Wx[C<t-1>,a<t-1>,x<t>]+bx) 
where X stands for F/O/U 门
"""

# 由于GRU更简单，一般比LSTM可以更加深层，但是一般还是LSTM更强大；

#%% 11 双向神经网络: Bidirectional RNN
# He said, "Teddy bears are on sale"
# He said, "Teddy Roosevelt was a great President!"
# 在判断Teddy是不是人名一部分的时候，只看之前的"He said"是不够的，还需要后面的信息；所以上面的单向RNN并不好用了
# 引入双向的RNN，除了正向的a<t-1>输入以外，还有个方向是接受a<t+1>, a<t+1> -> a<t>_reversed
# 由此构成一个Acyclic graph 无环图

#%% 12 DEEP RNNS
# 我们通常将多个RNN叠加起来，构建一个强大的DEEP RNN网络

"""
DEEP RNNS： 叠加多个RNN

x<1> -> a<1> -> a<2> -> a<...> -> y<1>
使用a[1]<0>表示第1层t=0步骤的元素；
其计算资源非常大，所以一般不会特别深，RNN之后可以加入全连接层
"""