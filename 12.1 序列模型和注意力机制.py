# 1 Basic model: Sequence to Sequence model

# Jane visite i'Afrique en septembre.
# x<1>, x<2>,  x<3>,    x<4>, x<5>
# Jane is visiting Africa in September.
# y<1>, y<2>, y<3>, y<4>, y<5>, y<6>

# (1) 构建一个RNN网络：encoder
# (2) 构建一个decoder：用语言模型生成后面的y
# 除了机器翻译，也可以用来描述图片：图片一个猫坐在椅子上；也可以用encoder-deconder结构
# 如使用预训练的Alex网络，然后去掉softmax层，最后一层4096作为编码，feed into the input as decoder nn 即可

#%% 2 选择最可能的句子
# language model: conditional model
# machine translation: encoder-> deconder(这部分和language model相似)
# 所以我们不希望随机按照分布生成翻译，而是求：argmax_yi[P(y<1>,...,y<t>)x),beam search通常用来解决这个问题
# 为何不使用贪心？下面例子说明了不考虑全局仅考虑局部最优不合适
# Jane is visiting Africa in September
# Jane is going to be visiting Africa in September.
# 第1个翻译更好，但是贪心算法认为going更常见，所以2可能认为更好；所以我们翻译的时候需要宏观考虑整个y的分布；

#%% 3 集束搜索算法 Beam search algorithm
# 算法流程
"""
可以复制beam width个解码部分中生成的部分（分别输出前width个），简化运算；
Step 1：对于我们的词汇表，我们将法语句子输入到编码网络中得到句子的编码，通过一个softmax层计算各个单词（词汇表中的所有单词）输出的概率值，
       通过设置集束宽度（beam width）的大小如3，我们则取前3个最大输出概率的单词，并保存起来。
Step 2：在第一步中得到的集束宽度的单词数，我们分别对第一步得到的每一个单词计算其与单词表中的所有单词组成词对的概率。并与第一步的概率相乘，
       得到第一和第二两个词对的概率。有3 × 10000 3\times 100003×10000个选择，（这里假设词汇表有10000个单词），
       最后再通过beam width大小选择前3个概率最大的输出对；
Step 3~Step T：与Step2的过程是相似的，直到遇到句尾符号结束。
"""

#%% 4 改进的集束搜索

# 对于集束搜索算法，我们的目标是最大化下面的概率：
# argmax_y P(y<1>,...y<t>|x)=argmax_y P(y<1>|x)*P(y<2>|y<1>,x)...
# 上面的得到的每一项一般都是很小的概率值，大量很小的概率值进行相乘，最后会得到更小的值，可能会造成数值下溢。所以在实践中，我们不会最大化上面这个公式的乘积，
# 而是取log值，变成log求和最大值，得到一个数值上更加稳定的算法，公式如下：
# argmax_y sum(logP(y<t>|x,y<1>...y<t-1>))
# 另外，我们还可以通过对上面的目标进行归一化，使其得到更好的效果。因为一般log函数都是负数，该函数倾向于翻译简短的句子，所以我们需要对此进行进行归一化处理。
# 相比直接除以输出单词长度的值，可以使用更加柔和的方式:在y<t>上加上一个指数alpha
# 如alpha=0.7，通过调整其大小获得更好的效果
# 1/ty^alpha * argmax_y sum(logP(y<t>|x,y<1>,...y<t-1>)), 这样的alpha并没有理论依据，但是实际效果较好；
# 通过上面的目标，选取得分最大的句子，即为我们的模型最后得到的输出结果。

# 集束搜索讨论：Beam width：B的选择，B越大考虑的情况越多，但是所需要进行的计算量也就相应的越大。
# 在常见的产品系统中，一般设置B = 10，而更大的值（如100，1000，…）则需要对应用的领域和场景进行选择。
# 相比于算法范畴中的搜索算法像BFS或者DFS这些精确的搜索算法，Beam Search 算法运行的速度很快，但是不能保证找到目标准确的最大值。

#%% 5 集束搜索的误差分析
# 集束搜索算法是一种近似搜索算法，也被称为启发式搜索算法。它的输出不能保证总是可能性最大的句子，因为其每一步中仅记录着Beam width为3或者10或者100种的可能的句子。
# 所以，如果我们的集束搜素算法出现错误了要怎么办呢？我们如何确定是算法出现了错误还是模型出现了错误呢？此时集束搜索算法的误差分析就显示出了作用。
# 例子：同样以法语句子的英文翻译为例子，我们人类对法语句子的翻译如中间的句子，而我们的模型输出的翻译如下面的句子。通过我们的模型，我们分别计算人类翻译的概率
# P ( y ∗ ∣ x )以及模型翻译的概率P(y^|x)，比较两个概率的大小，通过比较我们就能知道是因为Beam Search 算法的问题还是RNN模型的问题。如下图所示：
# if P(y*|x)>P(y^|x)的情况：Beam search算法选择了y^,但是y*却得到了更高的概率，所以Beam search算法出错了；
# if P(y*|x)<=P(y^|x)的情况:翻译结果y*比y^好，但rnn预测P(y*|x)<P(y^|x),所以这里是RNN模型出现错误；

# 在开发集上，对各个句子进行检测，得到每个句子对应的出错情况，那么根据整个开发集的上算法错误和模型错误的比例，就可以针对性地对二者之一进行改进和修正了。
"""
Human | Algorithm | P(y*|x) | P(y^|x) | At fault

...
"""

#%% 6 Bleu Score（评估机器翻译的效果）-> Bilingual evaluation understudy 双语评估替补
# Le chat est sur le tapis
# Ref 1: The cat is on the mat.
# Ref 2: There is a cat on the mat.
# MT output: The The The The The The The..
# MT output2: The cat the cat on the mat.
# 只要机器翻译的效果和任意一个人工翻译的结果相似，则他就会得到一个高的bleu分数
# Precesion:观察机器翻译的每一个词是否出现在参考中： 7/7 = 1
# Mofified Precision: 把每个单词的计分上限定义为他在参考句子中出现的最多次数。the在ref中出现了2次，所以2/7
# Bleu:不仅仅关注词，还考虑成对的单词（Bleu score on bigrams，相邻的两个词）： the cat/cat the/cat on...;
"""         count       countclip
the cat       2             1
cat the       1             0
cat on        1             1
on the        1             1 
the mat       1             1 
则最后是4/6
以此类推，我们即改良后的precision单个单词的是p1 = sum(countclip(unigram))/sum(count(unigram))
pn类似；推广到n即可，即可求n元词组的关系；
我们将其组合一下，得到最后的Bleu得分；
Pn = Bleu score on n-grams only;
Combined Bleu score: BP * exp(1/n * sum(p1,p2..pn)),其中bp为brevity penalty简短惩罚
BP = 1 if MT_output_length > reference_output_length
   = exp(1-MT_output_length/reference_output_length) otherwise
   
Bleu除了机器翻译，还可以评估很多生成文本的系统，如图片描述等等；
"""

#%% 7 注意力机制
# 机器翻译通过记忆编码器输入的文本部分，然后解码器 吐出来。但是实际翻译中，人工翻译的时候也是逐句逐段的翻译，很难一下子记忆全部的信息，
# 具体反映在Bleu score在短句子表现很好，但是长句子表现明显下降。所以我们通常会设置一个注意力机制，
# 来解决难以记录全部信息的问题，让机器实现逐步翻译；
"""
注意力模型的一个示例网络结构如上图所示。其中，底层是一个双向循环神经网络（BRNN），该网络中每个时间步的激活都包含前向传播和反向传播产生的激活：
# 使用t'表示原句子中的词；
a<t'> = (a-><t'>, a<-<t'>);
顶层是一个“多对多”结构的循环神经网络，
        y<1>
          |
s<0> -> s<1>
          |
          c<1>  
       /  |  \
  alpha<1,1> a<1,2> ... (注意力参数：告诉我们上下文有多少需要关注) 

我们保证注意力参数非负，且sumt'(alpha<1,t'>)=1,那么 c<1> = sumt'(alpha<1,t'>*a<t'>),其中a<t'>是t时刻的激活值输出值；
alpha<t,t'> = amount of attention y<t> should pay to a<t'>
现在问题就剩下，如何计算alpha<t,t'>
alpha<t,t'> = exp(e<t,t'>)/sumt'(e<t,t'>) 即使用softmax使其变成非0总和为1；
其中：e<t,t'> 可以用一个小的神经网络： s<t-1> -> |-| -> e<t,t'>
                                 a<t'>  -> |-|
来求得该整个注意力的权重；
该算法的一个欠缺点是时间复杂度为n(o3) ;
注意力机制的一个应用是为图片添加标题；
"""

#%% 8 语音识别
# 在语音识别任务中，输入是一段以时间为横轴的音频片段，输出是文本。
# 音频数据的常见预处理步骤是运行音频片段来生成一个声谱图，并将其作为特征。以前的语音识别系统通过语言学家人工设计的音素（Phonemes）来构建，
# 音素指的是一种语言中能区别两个词的最小语音单位。现在的端到端系统中，用深度学习就可以实现输入音频，直接输出文本。
# 对于训练基于深度学习的语音识别系统，大规模的数据集是必要的。学术研究中通常使用 3000 小时长度的音频数据，而商业应用则需要超过一万小时的数据。
# 语音识别系统可以用注意力模型来构建，一个简单的图例如下：
"""
在x输入上，输入不同时间帧；然后用注意力机制，逐步输出单词；
"""
# 用 CTC（Connectionist Temporal Classification）损失函数来做语音识别的效果也不错。由于输入是音频数据，使用 RNN 所建立的系统含有很多个时间步，
# 且输出数量往往小于输入。因此，不是每一个时间步都有对应的输出。CTC 允许 RNN 生成下图红字所示的输出，并将两个空白符（blank）中重复的字符折叠起来，再将空白符去掉，得到最终的输出文本。
"""
ttt___h_eee___space___qqq__...这样的输出也被视为正确的输出，最后输出The q...
最后把他中间重复的折叠即可
"""
#%% 9 触发词检测
# 触发词检测（Trigger Word Detection）常用于各种智能设备，通过约定的触发词可以语音唤醒设备。
# 使用 RNN 来实现触发词检测时，可以将触发词对应的序列的标签设置为“1”，而将其他的标签设置为“0”。
# 但是由于1肯定比0很多，所以可以将触发词之后一丢丢部分设置为1，增加1的比例；

#%% 10 结课和致谢
"""
感谢吴恩达，
感谢家人，
感谢爱人，
感谢我自己！
"""
