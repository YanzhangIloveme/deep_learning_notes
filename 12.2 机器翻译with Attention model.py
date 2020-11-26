# 欢迎来到本周的第一个编程作业！
# 您将构建一个神经机器翻译 (NMT) 模型，将人类可读日期 (“25th of June, 2009”) 翻译为机器可读日期 (“2009-06-25”). 您将使用注意模型执行此操作, 序列模型中最复杂的序列之一。
# 这个 notebook 是与NVIDIA的深度学习研究所共同制作的。
# 让我们加载此作业所需的所有包。

from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import numpy as np
import sys
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date


sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/Machine Translation')
from nmt_utils import *
import matplotlib.pyplot as plt


# 您将在此处构建的模型可用于从一种语言翻译到另一种语言, 例如从英语翻译成印地语。 但是，语言翻译需要大量数据集，并且通常需要使用 GPU 训练数天。
# 为了让您在不使用大量数据集的情况下尝试使用这些模型，我们将使用更简单的“日期转换”任务。 “date translation” task.
# 网络将输入以各种可能格式编写的日期 (例如：“the 29th of August 1958”, “03/30/1968”, “24 JUNE 1987”)
# 将它们转换为标准化的机器可读日期 (例如：“1958-08-29”, “1968-03-30”, “1987-06-24”). 我们将让网络学会以通用的机器可读格式输出日期YYYY-MM-DD.

#%% 1.1 数据集和数据处理

#   我们将在 10000 组人类可读日期，并且与之对应的，标准化的机器可读日期的数据集上训练模型。运行下面的单元价值数据集并且打印一些样例。
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print(dataset[:10])


# 已经加载了：
#
# dataset: 一个元组列表 (人类可读日期, 机器可读日期)。
# human_vocab: 一个python字典，将人类可读日期中使用的所有字符映射到整数值索引。
# machine_vocab: 一个python字典，将机器可读日期中使用的所有字符映射到整数值索引。这些索引不一定与 human_vocab 的索引一致。
# inv_machine_vocab: machine_vocab的逆字典，从索引到字符的映射。
# 让我们对数据进行预处理，将原始文本数据映射到索引值。
"""
我们使用：

  Tx=30 (我们假设人类可读日期的最大长度; 如果我们得到更长的输入，我们将截断它)

  Ty=10 (因为 “YYYY-MM-DD” 是 10 个字符长度).

"""
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
"""
你已经有:

X: 训练集中人类可读日期的处理版本, 其中每个字符都被它在 human_vocab 中映射该字符的索引替换。每个日期都使用特殊字符（）进一步填充。维度为 X.shape = (m, Tx)
Y: 训练集中机器可读日期的处理版本, 其中每个字符都被它在machine_vocab中映射的索引替换。 维度为 Y.shape = (m, Ty)。
Xoh: X 的 one-hot 版本, one-hot 中条目 “1” 的索引被映射到在human_vocab中对应字符。维度为 Xoh.shape = (m, Tx, len(human_vocab))
Yoh: Y 的 one-hot 版本, one-hot 中条目 “1” 的索引被映射到由于machine_vocab 中对应字符。维度为 Yoh.shape = (m, Tx, len(machine_vocab))。 
这里, len(machine_vocab) = 11 因为有 11 字符 (’-’ 以及 0-9).

"""

#%% 1.2 - 带注意力的神经机器翻译
#   如果你不得不将一本书的段落从法语翻译成英语, 你不会阅读整段，然后合上书并翻译。甚至在翻译过程中, 您将阅读/重新阅读并专注于与您所写的英语部分相对应的法文段落部分。
#   注意机制告诉神经机器翻译模型，它应该在任何一步都要有注意力。
"""
模型中有两个单独的 LSTM（见左图）。一个是图片底部的那个是 Bi-LSTM（双向LSTM）在 Attention 前，我们叫做 pre-attention Bi-LSTM。
图的顶部的 LSTM 在 Attention 后，我们叫做 post-attention LSTM。pre-attention Bi-LSTM 经历 T_xT 时间步；
post-attention LSTM 经历T_yT 时间步post-attention LSTM 通过 s⟨t⟩ , c⟨t⟩ 从一个时间步到下一个。在视频讲座中, 对于post-attention 
序列模型我们仅使用了基本的 RNN，状态被 RNN 输出激活捕获 s⟨t⟩。但是因为我们在这里使用 LSTM , LSTM 有输出激活 s⟨t⟩和隐藏单元状态 c⟨t⟩。
但是，与之前的文本生成示例（例如第1周的Dinosaurus）不同，在此模型中， post-activation LSTM 在时间 t tt 不会用具体生成的 y⟨ t − 1⟩
作为输入; 只需要 s⟨t⟩和 c⟨t⟩ 作为输入。
我们以这种方式设计了模型，因为（与相邻字符高度相关的语言生成不同） 在YYYY-MM-DD日期中，前一个字符与下一个字符之间的依赖性不强。

让我们实现这个模型。您将从实现两个功能开始： one_step_attention() 和 model()。

"""

# 1) one_step_attention():
# 在步骤t，给出Bi-Lstm的所有隐藏状态[a<1>,a<2>,...a<tx>]以及第二个lstm的先前隐藏状态S<t-1>
# one_step_attention计算注意力权重[a<t,1>,a<t,2>...a<t,Tx>]并输出上下文向量context<t> = sum(a<t,t'> * a<t'>)

# 2) model():
# 实现整个模型，首先根据输入执行Bi-listm返回[a<1>...a<Tx>]然后，调用one_step_attension()中的层Ty次，很重要一点是所有Ty的拷贝具有相同权重
# 也就是，不应该每次都重新设置权重，换句话，Ty步骤都共享权重。
# 1 定义层对象 （作为样本的全局变量）； 2 在传播输入时调用这些对象

# 将共享层定义为全局变量
repeator = RepeatVector(Tx) # repeate 30次
concatenator = Concatenate(axis=-1) # 在最后一维度聚合（37个字符）
densor1 = Dense(10, activation = "tanh") # 全连接
densor2 = Dense(1, activation = "relu") # softmax
activator = Activation(softmax, name='attention_weights') # 在这个 notebook 我们正在使用自定义的 softmax(axis = 1)
dotor = Dot(axes = 1)

#   现在您可以使用这些图层来实现 one_step_attention()。为了通过这些层之一传播Keras张量对象X，使用 layer(X) (或 layer([X,Y]) 如果它需要多个输入),
#   例如， densor（X）将通过上面定义的Dense（1）层传播 X。

def one_step_attention(a, s_prev):
    """
    执行一步 attention: 输出一个上下文向量，输出作为注意力权重的点积计算的上下文向量
    "alphas"  Bi-LSTM的 隐藏状态 "a"

    参数：
    a --  Bi-LSTM的输出隐藏状态 numpy-array 维度 (m, Tx, 2*n_a)
    s_prev -- (post-attention) LSTM的前一个隐藏状态, numpy-array 维度(m, n_s)

    返回：
    context -- 上下文向量, 下一个(post-attetion) LSTM 单元的输入
    """

    # 使用 repeator 重复 s_prev 维度 (m, Tx, n_s) 这样你就可以将它与所有隐藏状态"a" 连接起来。 (≈ 1 line)
    s_prev = repeator(s_prev)
    # 使用 concatenator 在最后一个轴上连接 a 和 s_prev (≈ 1 line)
    concat = concatenator([a, s_prev])
    # 使用 densor1 传入参数 concat, 通过一个小的全连接神经网络来计算“中间能量”变量 e。(≈1 lines)
    e = densor1(concat)
    # 使用 densor2 传入参数 e , 通过一个小的全连接神经网络来计算“能量”变量 energies。(≈1 lines)
    energies = densor2(e)
    # 使用 activator 传入参数 "energies" 计算注意力权重 "alphas" (≈ 1 line)
    alphas = activator(energies)
    # 使用 dotor 传入参数 "alphas" 和 "a" 计算下一个（(post-attention) LSTM 单元的上下文向量 (≈ 1 line)
    context = dotor([alphas, a])

    return context

#   在编写 model() 函数之后，您将能够检查 one_step_attention() 的预期输出。
#%% Model()函数
# 练习: 实现 model() 如图1.1和上文所述。 同样，我们已经定义了全局图层用于在 model（） 中共享权重。
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)

# 现在您可以在for循环中使用这些图层Ty次来生成输出，并且他们的参数不会重新初始化。您必须执行以下步骤：
# 1 将参数传入到Bi-LSTM
# 2 迭代 for t = 0...Ty-1
#           - 调用 one_step_attention() 使用【a<t,1>, a<t,2>,...,a<t,Tx>】和s<t-1>为参数，获取上下文向量context<t>
#           - 使用context<t>作为参数给post-attention LSTM单元，记得传入LSTM以前的隐藏状态S<t-1>和单元状态C<t-1>，
#             使用initial_state= [previous hidden state, previous cell state]，返回隐藏状态S<t>和新的单元状态C<t>
#           - 应用softm图层S<t>，来获得输出
#           - 通过将输出添加到输出列表保存输出
# 3 创建tensorflow.keras模型实例，应该有三个输入：（输入列表, S<0> 和 C<0> ）
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    参数:
    Tx -- 输入序列的长度
    Ty -- 输出序列的长度
    n_a -- Bi-LSTM的隐藏状态大小
    n_s -- post-attention LSTM的隐藏状态大小
    human_vocab_size -- python字典 "human_vocab" 的大小
    machine_vocab_size -- python字典 "machine_vocab" 的大小

    返回：
    model -- Keras 模型实例
    """
    # 定义模型的输入，维度 (Tx,)
    # 定义 s0 和 c0, 初始化解码器 LSTM 的隐藏状态，维度 (n_s,)
    X = Input(shape=(Tx, human_vocab_size)) # shape=(None, 30, 37)
    s0 = Input(shape=(n_s,), name='s0') #
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    # 初始化一个空的输出列表
    outputs = []
    # 第一步：定义 pre-attention Bi-LSTM。 记得使用 return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a * 2))(X)

    # 第二步：迭代 Ty 步
    for t in range(Ty):
        # 第二步.A: 执行一步注意机制，得到在 t 步的上下文向量 (≈ 1 line)
        context = one_step_attention(a,s)
        # 第二步.B: 使用 post-attention LSTM 单元得到新的 "context"
        # 别忘了使用： initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        # 第二步.C: 使用全连接层处理post-attention LSTM 的隐藏状态输出 (≈ 1 line)
        out = output_layer(s)
        # 第二步.D: 追加 "out" 到 "outputs" 列表 (≈ 1 line)
        outputs.append(out)

    # 第三步：创建模型实例，获取三个输入并返回输出列表。 (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

#%%
# 像往常一样，在Keras创建模型后，你需要编译它并定义模型使用的损失, 优化和评估指标。编译模型时，损失使用 categorical_crossentropy，
# 优化算法使用 Adam optimizer (learning rate = 0.005, beta1=0.9, beta2=0.999,decay=0.01),=0.999, decay = 0.01)，评估指标使用 ['accuracy']
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 最后一步式定义所有的输入和输出并训练模型：
# 你已经有包含训练样例的 X，维度 ( m = 10000 , Tx = 30)
# 你需要创建用0初始化的 s0 和 c0，用于初始的 post_activation_LSTM_cell。
# 给定的 model() ，你需要“输出”11个维度为（m，T_y）元素的列表，以便 outputs[i][0],...outputs[i][Ty]表示第 ith个训练样本(X[i])对应的
# 真实标签。大多数情况下第ith个训练样本中第jth个字符真正标签是outputs[i][j]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
#%% 加载已经训练好的模型：
model.load_weights('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/Machine Translation/models/model.h5')
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']
s0 = np.zeros((1, n_s))
c0 = np.zeros((1, n_s))
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))
#%%
"""

总结一下，return_sequences即表示，LSTM的输出h(t)，是输出最后一个timestep的h(t)，还是把所有timestep的h(t)都输出出来。
在实际应用中，关系到网络的应用场景是many-to-one还是many-to-many，非常重要。

接下来我们继续实验return_state
注意，输出是一个列表list，分别表示 - 最后一个time step的hidden state - 最后一个time step的hidden state（跟上面一样) - 
最后一个time step的cell state（注意就是上文中的c(t)）
可以看出，return_state就是控制LSTM中的c(t)输出与否；

需要记住的是：:

机器翻译模型可用于从一个序列映射到另一个序列。 它们不仅可用于翻译人类语言（如法语 - >英语），还可用于日期格式翻译等任务。
注意机制允许网络在生成输出的特定部分时关注输入的最相关部分。
使用注意机制的网络可以从长度为 Tx 的输入转换为长度为Ty 的输出，其Tx和Ty可以不同。
你可以将注意力权重 α⟨t , t′⟩可视化，查看在生成每个输出时网络正在关注什么。
您现在可以实现注意模型并使用它来学习从一个序列到另一个序列的复杂映射。
"""

