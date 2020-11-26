# Basic training loops：Solving machine learning problems

    # Obtain training data.
    # Define the model.
    # Define a loss function.
    # Run through the training data, calculating loss from the ideal value
    # Calculate gradients for that loss and use an optimizer to adjust the variables to fit the data.
    # Evaluate your results.

#%% 1 data
# Supervised learning uses inputs (usually denoted as x) and outputs (denoted y, often called labels).
# The goal is to learn from paired inputs and outputs so that you can prediect the value of an output from an input.

# Each input of your data, in TensorFlow, is almost always represented by a tensor, and is often a vector.
# In supervised training, the output (or value you'd like to predict) is also a tensor.

import tensorflow as tf

TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 1000

# A vector of random x values
x = tf.random.normal(shape=[NUM_EXAMPLES])

# Generate some noise
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# Calculate y
y = x * TRUE_W + TRUE_B + noise

# Plot all the data
import matplotlib.pyplot as plt

plt.scatter(x, y, c="b")
plt.show()
"""
Tensors are usually gathered together in batches, or groups of inputs and outputs stacked together. Batching can confer 
some training benefits and works well with accelerators and vectorized computation. Given how small this dataset is, 
you can treat the entire dataset as a single batch.
"""

#%% 2 Define the model

class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b

model = MyModel()
# List the variables tf.modules's built-in variable aggregation.
print("Variables:", model.variables)
# Verify the model works
assert model(3.0).numpy() == 15.0

#%% 3 Define a loss function

# A loss function measures how well the output of a model for a given input matches the target output.
# The goal is to minimize this difference during training. Define the standard L2 loss, also known as the "mean squared" error:
def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

plt.scatter(x, y, c="b")
plt.scatter(x, model(x), c="r")
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())

#%% 4 Define a training loop
# The training loop consists of repeatedly doing three tasks in order:

# Sending a batch of inputs through the model to generate outputs
# Calculating the loss by comparing the outputs to the output (or label)
# Using gradient tape to find the gradients
# Optimizing the variables with those gradients

import tensorflow as tf

def train(model, x, y, learning_rate):

  with tf.GradientTape() as t:
    # Trainabl
    # e variables are automatically tracked by GradientTape
    current_loss = loss(y, model(x))

  # Use GradientTape to calculate the gradients with respect to W and b
  dw, db = t.gradient(current_loss, [model.w, model.b])

  # Subtract the gradient scaled by the learning rate
  model.w.assign_sub(learning_rate * dw)
  model.b.assign_sub(learning_rate * db)

model = MyModel()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)

# Define a training loop
def training_loop(model, x, y):

  for epoch in epochs:
    # Update the model with the single giant batch
    train(model, x, y, learning_rate=0.1)

    # Track this before I update
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(y, model(x))

    print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
          (epoch, Ws[-1], bs[-1], current_loss))


print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" %
      (model.w, model.b, loss(y, model(x))))

# Do the training
training_loop(model, x, y)

# Plot it
plt.plot(epochs, Ws, "r",
         epochs, bs, "b")

plt.plot([TRUE_W] * len(epochs), "r--",
         [TRUE_B] * len(epochs), "b--")

plt.legend(["W", "b", "True W", "True b"])
plt.show()

#%% The same solution, but with Keras
class MyModelKeras(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x, **kwargs):
    return self.w * x + self.b

keras_model = MyModelKeras()

# Reuse the training loop with a Keras model
training_loop(keras_model, x, y)

# You can also save a checkpoint using Keras's built-in support
keras_model.save_weights("my_checkpoint")

#%% part2 Advanced Automatic Differentiation -1 Controlling gradient recording
# If you wish to stop recording gradients, you can use GradientTape.stop_recording() to temporarily suspend recording.
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  x_sq = x * x
  with t.stop_recording():
    y_sq = y * y
  z = x_sq + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])
# If you wish to start over entirely, use reset()

#%% 2 Stop gradient => tf.stop_gradient
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  y_sq = y**2
  z = x**2 + tf.stop_gradient(y_sq)

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

#%% 3 Custom gradients
# In some cases, you may want to control exactly how gradients are calculated rather than using the default. These situations include:

# There is no defined gradient for a new op you are writing.
# The default calculations are numerically unstable.
# You wish to cache an expensive computation from the forward pass.
# You want to modify a value (for example using: tf.clip_by_value, tf.math.round) without modifying the gradient.

# For writing a new op, you can use tf.RegisterGradient to set up your own
# For the latter three cases, you can use tf.custom_gradient.
# Here is an example that applies tf.clip_by_norm to the intermediate gradient.
@tf.custom_gradient
def clip_gradients(y):
  def backward(dy):
    return tf.clip_by_norm(dy, 0.5)
  return y, backward

v = tf.Variable(2.0)
with tf.GradientTape() as t:
  output = clip_gradients(v * v)
print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

#%% 3 Multiple tapes interact seamlessly. For example, here each tape watches a different set of tensors:

x0 = tf.constant(0.0)
x1 = tf.constant(0.0)

with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
  tape0.watch(x0)
  tape1.watch(x1)

  y0 = tf.math.sin(x0)
  y1 = tf.nn.sigmoid(x1)

  y = y0 + y1
  ys = tf.reduce_sum(y)

#%% 4 Higher-order gradients
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t2:
  with tf.GradientTape() as t1:
    y = x * x * x

  # Compute the gradient inside the outer `t2` context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t1.gradient(y, x)
d2y_dx2 = t2.gradient(dy_dx, x)

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 *

#%% 5 Jacobians
x = tf.linspace(-10.0, 10.0, 200+1)
delta = tf.Variable(0.0)

with tf.GradientTape() as tape:
  y = tf.nn.sigmoid(x+delta)

dy_dx = tape.jacobian(y, delta)