"""
TensorFlow 2.0中进行了多项更改，以使TensorFlow用户更加高效。 TensorFlow 2.0删除了冗余API ，使API更加一致（ 统一RNN ， 统一优化器 ），
并通过Eager执行更好地与Python运行时集成。

"""

#%% 1 API清理
"""

在TF 2.0中，许多API都已消失或移动 。其中的一些主要更改包括删除tf.app ， tf.flags和tf.logging以支持现在开放源代码的absl-py ，
重定位驻留在tf.contrib项目，并通过清理主tf.*名称空间将较少使用的函数移入tf.math子包中。某些API已被其2.0等效项取代tf.summary ， 
tf.keras.metrics和tf.keras.optimizers 。自动应用这些重命名的最简单方法是使用v2升级脚本 。
"""

#%% 2 急于执行
"""
TensorFlow 1.X要求用户通过调用tf.* API手动将抽象语法树 （图形）拼接在一起。然后，它要求用户通过将一组输出张量和输入张量传递给session.run()
调用来手动编译抽象语法树。 TensorFlow 2.0急切地执行（就像Python通常一样），在2.0中，图和会话应该感觉像实现细节。

急切执行的一个显着副产品是不再需要tf.control_dependencies() ，因为所有代码行tf.function顺序执行（在tf.function ，具有副作用的代码按写入的顺序执行）。

"""

#%% 3 没有更多的全局变量
"""
TensorFlow 1.X严重依赖隐式全局名称空间。当您调用tf.Variable() ，它将被放到默认图中，即使您丢失了指向它的Python变量，它也将保留在默认图中。然后，您可以恢复该tf.Variable ，

但tf.Variable是您知道创建该文件的名称。如果您无法控制变量的创建，则很难做到这一点。结果，各种机制激增，试图再次帮助用户找到其变量，并为框架寻找用户创建的变量：变量作用域，全局集合，诸如tf.get_global_step() ， tf.global_variables_initializer()类的帮助器方法，优化程序隐式计算所有可训练变量的梯度，依此类推。 TensorFlow 2.0消除了所有这些机制（ Variables 2.0 RFC ），转而使用默认机制：跟踪变量！如果您忘记了tf.Variable ，则会收集垃圾。

跟踪变量的要求为用户带来了一些额外的工作，但是对于Keras对象（请参见下文），负担最小化。"""


#%% 4 功能而不是会话
"""
session.run()调用几乎就像一个函数调用：指定输入和要调用的函数，然后获取一组输出。在TensorFlow 2.0中，您可以使用tf.function()装饰一个Python函数以将其标记为JIT编译，
以便TensorFlow将其作为单个图运行（ Functions 2.0 RFC ）。该机制使TensorFlow 2.0能够获得图形模式的所有优势：

性能：可以优化功能（节点修剪，内核融合等）
可移植性：可以导出/重新导入该函数（ SavedModel 2.0 RFC ），允许用户重用和共享模块化TensorFlow函数。
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
"""

#%% 5 Tensorflow 2.0 惯例

#%% 5.1 将代码重构为较小的函数
#%% 5.2 使用Keras图层和模型来管理变量

"""
原版本
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

keras 版本
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

Keras图层/模型继承自tf.train.Checkpointable并与@tf.function集成，这使得可以直接从Keras对象进行检查点或导出SavedModels。
您不一定必须使用.fit()的.fit() API来利用这些集成。

这是一个转移学习示例，演示了Keras如何使收集相关变量的子集变得容易。假设您正在使用共享主干训练多头模型：

"""
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# Train on primary dataset
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path1(x, training=True)
    loss = loss_fn_head1(prediction, y)
  # Simultaneously optimize trunk and head1 weights.
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# Fine-tune second head, reusing the trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path2(x, training=True)
    loss = loss_fn_head2(prediction, y)
  # Only optimize head2 weights, not trunk weights
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# You can publish just the trunk computation for other people to reuse.
tf.saved_model.save(trunk, output_path)

#%% 5.3 结合使用tf.data.Datasets和@ tf.function
"""
当遍历适合内存的训练数据时，请随时使用常规Python迭代。否则， tf.data.Dataset是从磁盘流式传输训练数据的最佳方法。
数据集是可迭代的（而不是迭代器） ，就像在Eager模式下的其他Python可迭代的一样工作。通过将代码包装在tf.function() ，
您可以充分利用数据集异步预取/流功能，该功能将Python迭代替换为使用AutoGraph的等效图形操作。

"""

@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      prediction = model(x, training=True)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 如果使用.fit() API，则不必担心数据集迭代。
    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(dataset)

#%% 5.4 利用带有Python控制流的AutoGraph
"""
AutoGraph提供了一种将依赖数据的控制流转换为等效于tf.cond和tf.while_loop图形模式的tf.while_loop 。

依赖于数据的控制流出现的一个常见地方是序列模型。 tf.keras.layers.RNN包装一个RNN单元，使您可以静态或动态展开重复。为了演示起见，您可以按以下方式重新实现动态展开：

"""

class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state

#%% 5.5 tf.metrics汇总数据，而tf.summary记录它们
"""
要记录摘要，请使用tf.summary.(scalar|histogram|...) ，然后使用上下文管理器将其重定向到作者。 
（如果省略上下文管理器，则什么也没有发生。）与TF 1.x不同，摘要直接发送给编写器；没有单独的“合并”操作，也没有单独的add_summary()调用，
这意味着必须在调用站点上提供step值。
"""
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)

# 要在将数据记录为摘要之前聚合数据，请使用tf.metrics 。指标是有状态的：当您调用.result()时，它们会累加值并返回累加结果。使用.reset_states()清除累积值。
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  loss = loss_fn(model(test_x, training=False), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)

# 通过将TensorBoard指向摘要日志目录来可视化生成的摘要：
#
# tensorboard --logdir /tmp/summaries

#%% 5.6  调试时使用tf.config.experimental_run_functions_eagerly（）
"""
在TensorFlow 2.0中，Eager执行使您可以逐步运行代码以检查形状，数据类型和值。某些API（例如tf.function ， tf.keras等）被设计为使用Graph执行，以提高性能和可移植性。
调试时，使用tf.config.experimental_run_functions_eagerly(True)在此代码内使用Eager执行。
"""

@tf.function
def f(x):
  if x > 0:
    import pdb
    pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)
f(tf.constant(1))


>>> f()
# -> x = x + 1
# (Pdb) l
#   6     @tf.function
#   7     def f(x):
#   8       if x > 0:
#   9         import pdb
#  10         pdb.set_trace()
#  11  ->     x = x + 1
#  12       return x
#  13
#  14     tf.config.experimental_run_functions_eagerly(True)
#  15     f(tf.constant(1))
# [EOF]

# 这也可以在Keras模型和其他支持Eager执行的API中使用：

class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      import pdb
      pdb.set_trace()
      return input_data // 2


tf.config.experimental_run_functions_eagerly(True)
model = CustomModel()
model(tf.constant([-2, -4]))

# >>> call()
# -> return input_data // 2
# (Pdb) l
#  10         if tf.reduce_mean(input_data) > 0:
#  11           return input_data
#  12         else:
#  13           import pdb
#  14           pdb.set_trace()
#  15  ->       return input_data // 2
#  16
#  17
#  18     tf.config.experimental_run_functions_eagerly(True)
#  19     model = CustomModel()
#  20     model(tf.constant([-2, -4]))

