#%% 1 Sequential Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# A Sequential model is not appropriate when:

# Your model has multiple inputs or multiple outputs
# Any of your layers has multiple inputs or multiple outputs
# You need to do layer sharing
# You want non-linear topology (e.g. a residual connection, a multi-branch model)

# You can also create a Sequential model incrementally via the add() method:
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

#%% Input/Output
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)

#%% Transfer learning with a Sequential model

# Transfer learning consists of freezing the bottom layers in a model and only training the top layers.
# Here are two common transfer learning blueprint involving Sequential models.
# First, let's say that you have a Sequential model, and you want to freeze all layers except the last one.
# In this case, you would simply iterate over model.layers and set layer.trainable = False on each layer, except the last one. Like this:
model = keras.Sequential([
    keras.Input(shape=(784))
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)

#%%
