#%% 1 Eager Execution (1.0中是计算图，2。0后是eager execution)

# 可以立即运算，不用像1.0中先构建计算图了。TF2.0的Eager可以使用GPU加速
import tensorflow as tf
import warnings
tf.executing_eagerly()
# 默认情况是开启eager execution的

#%% Now you can run TensorFlow operations and the results will return immediately:
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

#%% Tensor和numpy转化

# NumPy operations accept tf.Tensor arguments
# The TensorFlow tf.math operations convert Python objects and NumPy arrays to tf.Tensor objects
# The tf.Tensor.numpy method returns the object's value as a NumPy ndarray.

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# Broadcasting support
b = tf.add(a, 1)
print(b)

print(a*b)

import numpy as np
c = np.multiply(a, b)
print(c)
print(a.numpy())



#%% 2 Dynamic control flow

# A major benefit of eager execution is that all the functionality of the host language is available while your model is executing

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num) # 转化为tensor
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1
fizzbuzz(15)

#%% 3 Eager training-Computing gradients
# 自动求导，使用tf.GradientTape来追溯求导操作
# You can use tf.GradientTape to train and/or compute gradients in eager.
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)
# A particular tf.GradientTape can only compute one gradient;

#%% 4 Eager for Training a model

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3],activation='relu'),
    tf.keras.layers.Conv2D(32,[3,3],activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)

])

# Even without training, call the model and inspect the output in eager execution:

for images,labels in dataset.take(1):
  print("Logits: ", model(images[0:1]).numpy())

# While keras models have a builtin training loop (using the fit method), sometimes you need more customization.
# Here's an example, of a training loop implemented with eager
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []
# DIY Train_steps
def train_step(images, labels):
  with tf.GradientTape() as tape:
    logits = model(images, training=True) # 获取输出结果

    # Add asserts to check the shape of the output.
    tf.debugging.assert_equal(logits.shape, (32, 10))

    loss_value = loss_object(labels, logits)

  loss_history.append(loss_value.numpy().mean()) # 记录loss
  grads = tape.gradient(loss_value, model.trainable_variables) # 计算gradients
  optimizer.apply_gradients(zip(grads, model.trainable_variables)) # 将gradents应用在可训练的变量上

def train(epochs):
  for epoch in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset):
      train_step(images, labels)
    print ('Epoch {} finished'.format(epoch))

train(epochs = 3)
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

#%% 5 eager execution: Variables and optimizers
# tf.Variable objects store mutable tf.Tensor-like values accessed during training to make automatic differentiation easier.
# 这些variables可以被包装到layers或models中
# For example, the automatic differentiation example above can be rewritten:
class Linear(tf.keras.Model):
  def __init__(self):
    super(Linear, self).__init__() # 继承tf.keras.Model
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B


# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])
# Next:
#
# Create the model.
# The Derivatives of a loss function with respect to model parameters.
# A strategy for updating the variables based on the derivatives.
model = Linear()
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
steps = 300
for i in range(steps):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
#%% 6 Object-based saving
# A tf.keras.Model includes a convenient save_weights method allowing you to easily create a checkpoint:
model.save_weights('weights')
status = model.load_weights('weights')

# Using tf.train.Checkpoint you can take full control over this process.
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
x.assign(2.)   # Assign a new value to the variables and save.
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')
x.assign(11.)  # Change the variable after saving.
# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0

# To save and load models, tf.train.Checkpoint stores the internal state of objects, without requiring hidden variables.
# To record the state of a model, an optimizer, and a global step, pass them to a tf.train.Checkpoint:
# import os
# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
#   tf.keras.layers.GlobalAveragePooling2D(),
#   tf.keras.layers.Dense(10)
# ])
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# checkpoint_dir = 'path/to/model_dir'
# if not os.path.exists(checkpoint_dir):
#   os.makedirs(checkpoint_dir)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model)
#
# root.save(checkpoint_prefix)
# root.restore(tf.train.latest_checkpoint(checkpoint_dir))
#%% 7 Object-oriented metrics
# tf.keras.metrics are stored as objects. Update a metric by passing the new data to the callable,
# and retrieve the result using the tf.keras.metrics.result method, for example:

m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5

#%% 8 Summaries and TensorBoard
# TensorBoard is a visualization tool for understanding, debugging and optimizing the model training process.
# It uses summary events that are written while executing the program.

# You can use tf.summary to record summaries of variable in eager execution.
# For example, to record summaries of loss once every 100 training steps:

logdir = "./tb/"
writer = tf.summary.create_file_writer(logdir)

steps = 1000
with writer.as_default():  # or call writer.set_as_default() before the loop.
  for i in range(steps):
    step = i + 1
    # Calculate loss with your real train function.
    loss = 1 - 0.001 * step
    if step % 100 == 0:
      tf.summary.scalar('loss', loss, step=step)

#%% 9 Advanced automatic differentiation topics
# Dynamic models: tf.GradientTape can also be used in dynamic models.
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically tracked.
    # But to calculate a gradient from a tensor, you must `watch` it.
    tape.watch(init_x) # 确保init_x已经被记录了
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad) # Computes the sum of elements across dimensions of a tensor.
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value

# Custom gradients
# Custom gradients are an easy way to override gradients.
# Within the forward function, define the gradient with respect to the inputs, outputs, or intermediate results.
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x) # Return a Tensor with the same shape and contents as input.
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]   # Clips tensor values to a maximum L2-norm.
  return y, grad_fn

# Custom gradients are commonly used to provide a numerically stable gradient for a sequence of operations:
def log1pexp(x):
  return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

grad_log1pexp(tf.constant(0.)).numpy()
# However, x = 100 fails because of numerical instability.
grad_log1pexp(tf.constant(100.)).numpy()
# Here, the log1pexp function can be analytically simplified with a custom gradient.
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)
# As before, the gradient computation works fine at x = 0.
grad_log1pexp(tf.constant(0.)).numpy()
# And the gradient computation also works at x = 100.
grad_log1pexp(tf.constant(100.)).numpy()

#%% 10 Performance
# Computation is automatically offloaded to GPUs during eager execution.
# If you want control over where a computation runs you can enclose it in a tf.device('/gpu:0') block (or the CPU equivalent):
import time

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul can return before completing the matrix multiplication
  # (e.g., can return after enqueing the operation on a CUDA stream).
  # The x.numpy() call below will ensure that all enqueued operations
  # have completed (and will also copy the result to host memory,
  # so we're including a little more than just the matmul operation
  # time).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))
# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# Run on GPU, if available:
if tf.config.experimental.list_physical_devices("GPU"):
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
else:
  print("GPU: not found")

# A tf.Tensor object can be copied to a different device to execute its operations:
if tf.config.experimental.list_physical_devices("GPU"):
  x = tf.random.normal([10, 10])

  x_gpu0 = x.gpu()
  x_cpu = x.cpu()

  _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
  _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

#%% 11 Benchmarks
# For compute-heavy models, such as ResNet50 training on a GPU, eager execution performance is comparable to tf.function execution
# But this gap grows larger for models with less computation and
# there is work to be done for optimizing hot code paths for models with lots of small operations.


#%% 12 Work with functions
# While eager execution makes development and debugging more interactive,
# TensorFlow 1.x style graph execution has advantages for distributed training, performance optimizations, and production deployment.
# To bridge this gap, TensorFlow 2.0 introduces functions via the tf.function API. For more information, see the tf.function guide

#%% 补：tf.function编程
# Compiles a function into a callable TensorFlow graph.

"""
tf.function(
    func=None, input_signature=None, autograph=True, experimental_implements=None,
    experimental_autograph_options=None, experimental_relax_shapes=False,
    experimental_compile=None
)
"""
@tf.function
def f(x, y):
  return x ** 2 + y
x = tf.constant([2, 3])
y = tf.constant([3, -2])
f(x, y)

# func may use data-dependent control flow, including if, for, while break, continue and return statements:
@tf.function
def f(x):
  if tf.reduce_sum(x) > 0:
    return x * x
  else:
    return -x // 2
f(tf.constant(-2))

# func's closure may include tf.Tensor and tf.Variable objects:
@tf.function
def f():
  return x ** 2 + y
x = tf.constant([-2, -3])
y = tf.Variable([3, -2])
f()

# func may also use ops with side effects, such as tf.print, tf.Variable and others:
v = tf.Variable(1)
@tf.function
def f(x):
  for i in tf.range(x):
    v.assign_add(i)
f(3)
v

#%% Key point
# Key Point: Any Python side-effects (appending to a list, printing with print, etc) will only happen once, when func is traced.
# To have side-effects executed into your tf.function they need to be written as TF ops:
l = []
@tf.function
def f(x):
  for i in x:
    l.append(i + 1)    # Caution! Will only happen once when tracing
f(tf.constant([1, 2, 3]))
l
# Instead, use TensorFlow collections like tf.TensorArray:
@tf.function
def f(x):
  ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  for i in range(len(x)):
    ta = ta.write(i, x[i] + 1)
  return ta.stack()
f(tf.constant([1, 2, 3]))

@tf.function
def f(x):
  return x + 1
isinstance(f.get_concrete_function(1).graph, tf.Graph)

# Caution: Passing python scalars or lists as arguments to tf.function will always build a new graph.
# To avoid this, pass numeric arguments as Tensors whenever possible:
@tf.function
def f(x):
  return tf.abs(x)
f1 = f.get_concrete_function(1)
f2 = f.get_concrete_function(2)  # Slow - builds new graph
f1 is f2

f1 = f.get_concrete_function(tf.constant(1))
f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1
f1 is f2

@tf.function
def f(x):
  return x + 1
vector = tf.constant([1.0, 1.0])
matrix = tf.constant([[3.0]])
f.get_concrete_function(vector) is f.get_concrete_function(matrix)
