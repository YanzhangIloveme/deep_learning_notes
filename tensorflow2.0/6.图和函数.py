# In this guide you'll see the core of how TensorFlow allows you to make simple changes to your code to get graphs,
# and how they are stored and represented, and how you can use them to accelerate and export your models.

#%% 1 what are graphs?

# In the previous three guides, you have seen TensorFlow running eagerly. This means TensorFlow operations are executed by Python,
# operation by operation, and returning results back to Python. Eager TensorFlow takes advantage of GPUs,
# allowing you to place variables, tensors, and even operations on GPUs and TPUs. It is also easy to debug.

# However, running TensorFlow op-by-op in Python prevents a host of accelerations otherwise available.
# If you can extract tensor computations from Python, you can make them into a graph.

# Graphs are data structures that contain a set of tf. Operation objects, which represent units of computation;
# and tf. Tensor objects, which represent the units of data that flow between operations. They are defined in a tf. Graph context.
# Since these graphs are data structures, they can be saved, run, and restored all without the original Python code.


# With a graph, you have a great deal of flexibility. You can use your TensorFlow graph in environments that don't have a Python interpreter,
# like mobile applications, embedded devices, and backend servers. TensorFlow uses graphs as the format for saved models when it exports them from Python.
# Graphs are also easily optimized, allowing the compiler to do transformations like:
# Statically infer the value of tensors by folding constant nodes in your computation ("constant folding").
# Separate sub-parts of a computation that are independent and split them between threads or devices.
# Simplify arithmetic operations by eliminating common subexpressions.

# 简而言之，图形是非常有用的，让你的TensorFlow运行快，并行运行，并在多个设备上高效运行。
# 但是，您仍希望为了方便起见在 Python 中定义我们的机器学习模型（或其他计算），然后当您需要时自动构造图形。

#%% 2 跟踪图
# 在 TensorFlow 中创建图形的方式是使用tf.function，无论是直接调用还是修饰器。

import tensorflow as tf
import timeit
from datetime import datetime

# Define a Python function
def function_to_get_faster(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Create a `Function` object that contains a graph
a_function_that_uses_a_graph = tf.function(function_to_get_faster)
# Make some tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)
# It just works!
a_function_that_uses_a_graph(x1, y1, b1).numpy()

#%%

# tf.function-ed functions are Python callables that work the same as their Python values. They have a particular class (),
# but to you they act just as the non-traced version.python.eager.def_function.Function
# tf.function recursive traces any Python function it calls.
def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Use the decorator
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)
# Note that the callable will create a graph that
# includes inner_function() as well as outer_function()
outer_function(tf.constant([[1.0, 2.0]])).numpy()

#%% 3 Flow control and side effects
# Flow control and loops are converted to TensorFlow via tf.autograph by default.
# Autograph use a combination of of methods, including noting loop constructs, unrolling, and ASTsing.

def my_function(x):
  if tf.reduce_sum(x) <= 1:
    return x * x
  else:
    return x-1

a_function = tf.function(my_function)

print("First branch, with graph:", a_function(tf.constant(1.0)).numpy())
print("Second branch, with graph:", a_function(tf.constant([5.0, 5.0])).numpy())
# Don't read the output too carefully.
print(tf.autograph.to_code(my_function))

#%% 4 Seeing the speed up
# Just wrapping a tensor-using function in tf.function does not automatically speed up your code.
# 对于复杂的计算，图形可以提供显著的加速。这是因为图形减少了 Python 到设备的通信并执行一些加速。

# Create an oveerride model to classify pictures
class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

input_data = tf.random.uniform([60, 28, 28])
eager_model = SequentialModel()
graph_model = tf.function(eager_model)

print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=10000))
print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))

#%% 5 多态函数

# 跟踪函数时，将创建多态对象。多态函数是 Python 可调用的，它封装了一个 API 后面的几个具体函数图。Function
# Conceptually, then:
#
#   A tf.Graph is the raw, portable data structure describing a computation
#   A Function is a caching, tracing, dispatcher over ConcreteFunctions
#   A ConcreteFunction is an eager-compatible wrapper around a graph that lets you execute the graph from Python
# You can inspect a_function, which is the result of calling tf.function on the Python function my_function.
# In this example, calling a_function with three kinds of arguments results in three different concrete functions.
print(a_function)

print("Calling a `Function`:")
print("Int:", a_function(tf.constant(2)))
print("Float:", a_function(tf.constant(2.0)))
print("Rank-1 tensor of floats", a_function(tf.constant([2.0, 2.0, 2.0])))


# Get the concrete function that works on floats
print("Inspecting concrete functions")
print("Concrete function for float:")
print(a_function.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.float32)))
print("Concrete function for tensor of floats:")
print(a_function.get_concrete_function(tf.constant([2.0, 2.0, 2.0])))


# Concrete functions are callable
# Note: You won't normally do this, but instead just call the containing `Function`
cf = a_function.get_concrete_function(tf.constant(2))
print("Directly calling a concrete function:", cf(tf.constant(2)))
#%% 6 Reverting to eager execution

# You may find yourself looking at long stack traces, specially ones that refer to tf.Graph or with tf.Graph().as_default().
# This means you are likely running in a graph context. Core functions in TensorFlow use graph contexts, such as Keras's model.fit().

# Here are ways you can make sure you are running eagerly:
#
# Call models and layers directly as callables
# When using Keras compile/fit, at compile time use model.compile(run_eagerly=True)
# Set global execution mode via tf.config.run_functions_eagerly(True)

# Using run_eagerly=True
# Define an identity layer with an eager side effect
class EagerLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(EagerLayer, self).__init__(**kwargs)
    # Do some kind of initialization here

  def call(self, inputs):
    print("\nCurrently running eagerly", str(datetime.now()))
    return inputs

# Create an override model to classify pictures, adding the custom layer
class SequentialModel(tf.keras.Model):
  def __init__(self):
    super(SequentialModel, self).__init__()
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)
    self.eager = EagerLayer()

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return self.eager(x)

# Create an instance of this model
model = SequentialModel()

# Generate some nonsense pictures and labels
input_data = tf.random.uniform([60, 28, 28])
labels = tf.random.uniform([60])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# First, compile the model without eager.
model.compile(run_eagerly=False, loss=loss_fn)
# Now, call fit and see that the function is traced (twice) and then the eager effect never runs again.
model.fit(input_data, labels, epochs=3)

# If you run even a single epoch in eager, however, you can see the eager side effect twice.
print("Running eagerly")
# When compiling the model, set it to run eagerly
model.compile(run_eagerly=True, loss=loss_fn)

model.fit(input_data, labels, epochs=1)


#%% Using run_functions_eagerly

# You can also globally set everything to run eagerly. This is a switch that bypasses the polymorphic function's traced functions and calls the original function directly. You can use this for debugging.

# Now, globally set everything to run eagerly
tf.config.run_functions_eagerly(True)
print("Run all functions eagerly.")

# Create a polymorphic function
polymorphic_function = tf.function(model)

print("Tracing")
# This does, in fact, trace the function
print(polymorphic_function.get_concrete_function(input_data))

print("\nCalling twice eagerly")
# When you run the function again, you will see the side effect
# twice, as the function is running eagerly.
result = polymorphic_function(input_data)
result = polymorphic_function(input_data)

# Don't forget to set it back when you are done
tf.config.experimental_run_functions_eagerly(False)


#%% 7 Tracing and performance


# Use @tf.function decorator
@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!")  # This eager
  return x * x + tf.constant(2)

# This is traced the first time
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect
print(a_function_with_python_side_effect(tf.constant(3)))

# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))