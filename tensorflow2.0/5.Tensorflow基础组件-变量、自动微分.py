#%% TensorFlow 变量是用于表示程序处理的共享持久状态的推荐方法。本指南介绍在 TensorFlow 中如何创建、更新和管理 tf.Variable 的实例。

#变量通过 tf.Variable 类进行创建和跟踪。tf.Variable 表示张量，对它执行运算可以改变其值。
# 利用特定运算可以读取和修改此张量的值。更高级的库（如 tf.keras）使用 tf.Variable 来存储模型参数。

#%% 1 设置
# 本笔记本讨论变量布局。如果要查看变量位于哪一个设备上，请取消注释这一行代码。
import tensorflow as tf

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)


#%% 2 创建变量
# 要创建变量，请提供一个初始值。tf.Variable 与初始值的 dtype 相同。

my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

# 变量与张量的定义方式和操作行为都十分相似，实际上，它们都是 tf.Tensor 支持的一种数据结构。与张量类似，变量也有 dtype 和形状，并且可以导出至 NumPy。
# 大部分张量运算在变量上也可以按预期运行，不过变量无法重构形状。
print("A variable:",my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, ([1,4])))

# 如上所述，变量由张量提供支持。您可以使用 tf.Variable.assign 重新分配张量。调用 assign（通常）不会分配新张量，而会重用现有张量的内存。
a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable:
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# 如果在运算中像使用张量一样使用变量，那么通常会对支持张量执行运算。
# 从现有变量创建新变量会复制支持张量。两个变量不能共享同一内存空间。
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]
#%% 3 生命周期、命名和监视
# 在基于 Python 的 TensorFlow 中，tf.Variable 实例与其他 Python 对象的生命周期相同。如果没有对变量的引用，则会自动将其解除分配。
# 为了便于跟踪和调试，您还可以为变量命名。两个变量可以使用相同的名称。
# Create a and b; they have the same value but are backed by different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")
print(a == b)

# 保存和加载模型时会保留变量名。默认情况下，模型中的变量会自动获得唯一变量名，所以除非您希望自行命名，否则不必多此一举。
# 虽然变量对微分很重要，但某些变量不需要进行微分。在创建时，通过将 trainable 设置为 False 可以关闭梯度。例如，训练计步器就是一个不需要梯度的变量。
step_counter = tf.Variable(1, name='Steps', trainable=False)

#%% 4 放置变量和张量
# 为了提高性能，TensorFlow 会尝试将张量和变量放在与其 dtype 兼容的最快设备上。这意味着如果有 GPU，那么大部分变量都会放置在 GPU 上。
# 不过，我们可以重写变量的位置。在以下代码段中，即使存在可用的 GPU，我们也可以将一个浮点张量和一个变量放置在 CPU 上。
# 通过打开设备分配日志记录（参阅设置），可以查看变量的所在位置。
# 注：虽然可以手动放置变量，但使用分布策略是一种可优化计算的更便捷且可扩展的方式。
with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)

# 您可以将变量或张量的位置设置在一个设备上，然后在另一个设备上执行计算。但这样会产生延迟，因为需要在两个设备之间复制数据。
# 不过，如果您有多个 GPU 工作进程，但希望变量只有一个副本，则可以这样做。
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)

# 注：由于 tf.config.set_soft_device_placement 默认处于打开状态，所以，即使在没有 GPU 的设备上运行此代码，它也会运行，只不过乘法步骤在 CPU 上执行。

#%% Part 2 自动微分和梯度带 - 梯度带
# 在上一个教程中，我们介绍了 "张量"（Tensor）及其操作。本教程涉及自动微分（automatic differentitation），它是优化机器学习模型的关键技巧之一。

# TensorFlow 为自动微分提供了 tf.GradientTape API ，根据某个函数的输入变量来计算它的导数。
# Tensorflow 会把 'tf.GradientTape' 上下文中执行的所有操作都记录在一个磁带上 ("tape")。 然后基于这个磁带和每次操作产生的导数，
# 用反向微分法（"reverse mode differentiation"）来计算这些被“记录在案”的函数的导数。
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)


# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0

# 你也可以使用 tf.GradientTape 上下文计算过程产生的中间结果来求取导数。

x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

#%%  默认情况下，调用 GradientTape.gradient() 方法时， GradientTape 占用的资源会立即得到释放。通过创建一个持久的梯度带，可以计算同个函数的多个导数。
# 这样在磁带对象被垃圾回收时，就可以多次调用 'gradient()' 方法。例如：
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape


#%% 2 记录控制流
# 由于磁带会记录所有执行的操作，Python 控制流（如使用 if 和 while 的代码段）自然得到了处理。
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

#%% 3 高阶导数

# 在 'GradientTape' 上下文管理器中记录的操作会用于自动微分。如果导数是在上下文中计算的，导数的函数也会被记录下来。
# 因此，同个 API 可以用于高阶导数。例如：
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0
with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0





