# To do machine learning TensorFlow, you are likely to need to define, save, and restore a model.
# A model is, abstractly:
# A function that computes something on tensors (a forward pass)
# Some variables that can be updated in response to training

# In this guide, you will go below the surface of Keras to see how TensorFlow models are defined.
# This looks at how TensorFlow collects variables and models, as well as how they are saved and restored.

import tensorflow as tf
from datetime import datetime

#%% 1 Defining models and layers in TensorFlow
# Most models are made of layers. Layers are functions with a known mathematical structure that can be reused and have trainable variables.
# In TensorFlow, most high-level implementations of layers and models, such as Keras or Sonnet,are built on the same foundational class: tf.Module.
# Here's an example of a very simple tf.Module that operates on a scalar tensor:
class SimpleModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.a_variable = tf.Variable(5.0, name="train_me")
    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
  def __call__(self, x):
    return self.a_variable * x + self.non_trainable_variable


simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))

# Modules and, by extension, layers are deep-learning terminology for "objects": They have internal state, and methods that use that state.
# You can set the trainability of variables on and off for any reason, including freezing layers and variables during fine-tuning.

# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)

#%% This is an example of a two-layer linear layer model made out of modules.

class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

# And then the complete model, which makes two layer instances and applies them.

class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

# tf.Module instances will automatically collect, recusively, any tf.Variable or tf.Module instances assigned to it.
# This allows you to manage collections oftf.Modules with a single model instance, and save and load whole models.

#%% 2 Waiting to create variables
# You may have noticed here that you have to define both input and output sizes to the layer. This is so the w variable has a known shape and can be allocated.

class FlexibleDenseModule(tf.Module):
  # Note: No need for `in+features`
  def __init__(self, out_features, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.out_features = out_features

  def __call__(self, x):
    # Create variables on first call.
    if not self.is_built:
      self.w = tf.Variable(
        tf.random.normal([x.shape[-1], self.out_features]), name='w')
      self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
      self.is_built = True

    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

# Used in a module
class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = FlexibleDenseModule(out_features=3)
    self.dense_2 = FlexibleDenseModule(out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))


#%% 3 保存weights
# You can save a tf.Module as both a checkpoint and a SavedModel.
# chkp_path = "my_checkpoint"
# checkpoint = tf.train.Checkpoint(model=my_model)
# checkpoint.write(chkp_path)
# checkpoint.write(chkp_path)

# new_model = MySequentialModule()
# new_checkpoint = tf.train.Checkpoint(model=new_model)
# new_checkpoint.restore("my_checkpoint")
#
# # Should be the same result as above
# new_model(tf.constant([[2.0, 2.0, 2.0]]))

#%% 4 Saving functions

class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  @tf.function
  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model with a graph!
my_model = MySequentialModule(name="the_model")

print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))

# You can visualize the graph by tracing it within a TensorBoard summary.
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MySequentialModule()

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])))
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)

#%% 5 Creating a SavedModel

# The recommended way of sharing completely trained models is to use SavedModel.
# SavedModel contains both a collection of functions and a collection of weights.
# tf.saved_model.save(my_model, "the_saved_model")

#%% 6 Keras models and layers
# tf.keras.layers.Layer is the base class of all Keras layers, and it inherits from tf.Module.
# You can convert a module into a Keras layer just by swapping out the parent, and then changing __call__ to call:

class MyDense(tf.keras.layers.Layer):
  # Adding **kwargs to support base Keras layer arguemnts
  def __init__(self, in_features, out_features, **kwargs):
    super().__init__(**kwargs)

    # This will soon move to the build step; see below
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def call(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

simple_layer = MyDense(name="simple", in_features=3, out_features=3)

#%% BUild steps
# Keras layers come with an extra lifecycle step that allows you more flexibility in how you define your layers.
# This is defined in the build() function.

# build is called exactly once, and it is called with the shape of the input. It's usually used to create variables (weights).

class FlexibleDense(tf.keras.layers.Layer):
  # Note the added `**kwargs`, as Keras supports many arguments
  def __init__(self, out_features, **kwargs):
    super().__init__(**kwargs)
    self.out_features = out_features

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.w = tf.Variable(
      tf.random.normal([input_shape[-1], self.out_features]), name='w')
    self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.matmul(inputs, self.w) + self.b

# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)

print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))
#%% Keras models

class MySequentialModel(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.dense_1 = FlexibleDense(out_features=3)
    self.dense_2 = FlexibleDense(out_features=2)
  def call(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a Keras model!
my_sequential_model = MySequentialModel(name="the_model")

# Call it on a tensor, with random results
print("Model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))
