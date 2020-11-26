import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Keras 函数式 API 是一种比 tf.keras.Sequential API 更加灵活的模型创建方式。函数式 API 可以处理具有非线性拓扑的模型、具有共享层的模型，以及具有多个输入或输出的模型。
# 深度学习模型通常是层的有向无环图 (DAG)。因此，函数式 API 是构建层计算图的一种方式。
# 如构建input（784）-> dense(64) -> dense(64) -> dense(10)

inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64,activation='relu')(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs = inputs, outputs=outputs, name='minst_model')
model.summary()

#%% 1 使用相同的层计算图定义多个模型
# 在函数式 API 中，模型是通过在层计算图中指定其输入和输出来创建的。这意味着可以使用单个层计算图来生成多个模型。
# 在下面的示例中，您将使用相同的层堆栈来实例化两个模型：能够将图像输入转换为 16 维向量的 encoder 模型，以及用于训练的端到端 autoencoder 模型。

encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
# 在上例中，解码架构与编码架构严格对称，因此输出形状与输入形状 (28, 28, 1) 相同。

#%% 2 处理复杂的计算图拓扑
# 具有多个输入和输出的模型:函数式 API 使处理多个输入和输出变得容易。而这无法使用 Sequential API 处理。
# 例如，如果您要构建一个系统，该系统按照优先级对自定义问题工单进行排序，然后将工单传送到正确的部门，则此模型将具有三个输入：

# 工单标题（文本输入），
# 工单的文本正文（文本输入），以及
# 用户添加的任何标签（分类输入）
# 此模型将具有两个输出：
# (1) 介于 0 和 1 之间的优先级分数（标量 Sigmoid 输出），以及(2) 应该处理工单的部门（部门范围内的 Softmax 输出）。
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)
# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])
# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)
# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# 编译此模型时，可以为每个输出分配不同的损失。甚至可以为每个损失分配不同的权重，以调整其对总训练损失的贡献。
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)

# 由于输出层具有不同的名称，您还可以像下面这样指定损失：

# model.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss={
#         "priority": keras.losses.BinaryCrossentropy(from_logits=True),
#         "department": keras.losses.CategoricalCrossentropy(from_logits=True),
#     },
#     loss_weights=[1.0, 0.2],
# )

# 通过传递输入和目标的 NumPy 数组列表来训练模型：

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)

#%% 3 小 ResNet 模型
# 除了具有多个输入和输出的模型外，函数式 API 还使处理非线性连接拓扑（这些模型的层没有按顺序连接）变得容易。这是 Sequential API 无法处理的。
inputs = keras.Input(shape=(32,32,3), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])
x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()

#%% 4 共享层
# 函数式 API 的另一个很好的用途是使用共享层的模型。共享层是在同一个模型中多次重用的层实例，它们会学习与层计算图中的多个路径相对应的特征。
"""
共享层通常用于对来自相似空间（例如，两个具有相似词汇的不同文本）的输入进行编码。它们可以实现在这些不同的输入之间共享信息
以及在更少的数据上训练这种模型。如果在其中的一个输入中看到了一个给定单词，那么将有利于处理通过共享层的所有输入。
"""

# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

#%% 5 提取和重用层计算图中的节点
"""
由于要处理的层计算图是静态数据结构，可以对其进行访问和检查。而这就是将函数式模型绘制为图像的方式。
这也意味着您可以访问中间层的激活函数（计算图中的“节点”）并在其他地方重用它们，这对于特征提取之类的操作十分有用。
让我们来看一个例子。下面是一个 VGG19 模型，其权重已在 ImageNet 上进行了预训练
"""
vgg19 = tf.keras.applications.VGG19()
# 下面是通过查询计算图数据结构获得的模型的中间激活：
features_list = [layer.output for layer in vgg19.layers]
# 使用以下特征来创建新的特征提取模型，该模型会返回中间层激活的值：
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)

#%% 6 使用自定义层扩展 API
"""
tf.keras 包含了各种内置层，例如：

卷积层：Conv1D、Conv2D、Conv3D、Conv2DTranspose
池化层：MaxPooling1D、MaxPooling2D、MaxPooling3D、AveragePooling1D
RNN 层：GRU、LSTM、ConvLSTM2D
BatchNormalization、Dropout、Embedding 等

但是，如果找不到所需内容，可以通过创建您自己的层来方便地扩展 API。所有层都会子类化 Layer 类并实现下列方法：
call 方法，用于指定由层完成的计算。
build 方法，用于创建层的权重（这只是一种样式约定，因为您也可以在 __init__ 中创建权重）。

以下是 tf.keras.layers.Dense 的基本实现：
"""
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        # 创建权重
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        # 计算逻辑
        return tf.matmul(inputs, self.w) + self.b

inputs = layers.Input(shape=(4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
model.summary()
#%% 为了在您的自定义层中支持序列化，请定义一个 get_config 方法，返回层实例的构造函数参数：
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}

inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
config = model.get_config()

# 但是，当构建不容易表示为有向无环的层计算图的模型时，模型子类化会提供更大的灵活性。例如，您无法使用函数式 API 来实现 Tree-RNN，而必须直接子类化 Model 类。

new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
# 您也可以选择实现 from_config(cls, config) 类方法，该方法用于在给定其配置字典的情况下重新创建层实例。from_config 的默认实现如下：
# def from_config(cls, config):   return cls(**config)

