from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/RNN')
from shakespeare_utils import *
import io

#%% 为了节省时间，我们已经为莎士比亚诗集《十四行诗》模型训练了1000代，让我们再训练一下这个模型
# 当它完成了一代的训练——这也需要几分钟——你可以运行generate_output
# 它会提示你输入(小于40个字符)。这首诗将从你的句子开始
# print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
# model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
generate_output() #博主在这里输入hello


#%% 用LSTM网络即兴独奏爵士乐
# 现在我们先来加载库，其中，music21可能不在你的环境内，你需要在命令行中执行pip install msgpack以及pip install music21来获取。
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import IPython
import sys
sys.path.append('/Users/zhangyan/Desktop/旗云恒基/01.知识库/深度学习/datasets/RNN')

from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

n_a = 64
reshapor = Reshape((1, 78))                        #2.B
LSTM_cell = LSTM(n_a, return_state = True)        #2.C
densor = Dense(n_values, activation='softmax')    #2.D
x = Lambda(lambda x: X[:,t,:])(X)
a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])


def djmodel(Tx, n_a, n_values):
    """
    实现这个模型

    参数：
        Tx -- 语料库的长度
        n_a -- 激活值的数量
        n_values -- 音乐数据中唯一数据的数量

    返回：
        model -- Keras模型实体
    """
    # 定义输入数据的维度
    X = Input((Tx, n_values))

    # 定义a0, 初始化隐藏状态
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0

    # 第一步：创建一个空的outputs列表来保存LSTM的所有时间步的输出。
    outputs = []

    # 第二步：循环
    for t in range(Tx):
        ## 2.A：从X中选择第“t”个时间步向量
        x = Lambda(lambda x: X[:, t, :])(X)

        ## 2.B：使用reshapor来对x进行重构为(1, n_values)
        x = reshapor(x)

        ## 2.C：单步传播
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        ## 2.D：使用densor()应用于LSTM_Cell的隐藏状态输出
        out = densor(a)

        ## 2.E：把预测值添加到"outputs"列表中
        outputs.append(out)

    # 第三步：创建模型实体
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model


model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

# 编译模型，我们使用Adam优化器与分类熵损失。
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 初始化a0和c0，使LSTM的初始状态为零。
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)
