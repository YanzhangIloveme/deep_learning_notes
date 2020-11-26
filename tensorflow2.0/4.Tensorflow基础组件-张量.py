import tensorflow as tf
import numpy as np

# 张量是具有统一类型（称为 dtype）的多维数组。您可以在 tf.dtypes.DType 中查看所有支持的 dtypes。
# 如果您熟悉 NumPy，就会知道张量与 np.arrays 有一定的相似性。
# 就像 Python 数值和字符串一样，所有张量都是不可变的：永远无法更新张量的内容，只能创建新的张量。

#%% 1 基础知识

# 我们来创建一些基本张量。
# 下面是一个“标量”（或称“0 秩”张量）。标量包含单个值，但没有“轴”。
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# “向量”（或称“1 秩”张量）就像一个值的列表。向量有 1 个轴：
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
# “矩阵”（或称“2 秩”张量）有 2 个轴：
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
# 张量的轴可能更多，下面是一个包含 3 个轴的张量：
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])

print(rank_3_tensor)

# 通过使用 np.array 或 tensor.numpy 方法，您可以将张量转换为 NumPy 数组：
np.array(rank_2_tensor)
rank_2_tensor.numpy()

# 张量通常包含浮点型和整型数据，但是还有许多其他数据类型，包括：复杂的数值/字符串
# tf.Tensor 基类要求张量是“矩形”——也就是说，每个轴上的每一个元素大小相同。但是，张量有可以处理不同形状的特殊类型。
# 不规则张量（参阅下文中的 RaggedTensor）
# 稀疏张量（参阅下文中的 SparseTensor）
# 我们可以对张量执行基本数学运算，包括加法、逐元素乘法和矩阵乘法运算。
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
# 各种运算 (op) 都可以使用张量。
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))
#%% 2 形状简介
# 张量有形状。下面是几个相关术语：（张量和 tf.TensorShape 对象提供了方便的属性来访问：）
# 形状：张量的每个维度的长度（元素数量）。
# 秩：张量的维度数量。标量的秩为 0，向量的秩为 1，矩阵的秩为 2。
# 轴或维度：张量的一个特殊维度。
# 大小：张量的总项数，即乘积形状向量
rank_4_tensor = tf.zeros([3, 2, 4, 5])
# 4秩张量，形状：[3，2，4，5]
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# 虽然通常用索引来指代轴，但是您始终要记住每个轴的含义。轴一般按照从全局到局部的顺序进行排序：
# 首先是批次轴，随后是空间维度，最后是每个位置的特征。这样，在内存中，特征向量就会位于连续的区域。
#%% 3 索引
# 单轴索引
# TensorFlow 遵循标准 Python 索引规则（类似于在 Python 中为列表或字符串编制索引）以及 NumPy 索引的基本规则。
# 索引从 0 开始编制/负索引表示按倒序编制索引/冒号 : 用于切片 start:stop:step
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
# 使用标量编制索引会移除维度：
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
# 使用 : 切片编制索引会保留维度：
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# 多轴索引
# 更高秩的张量通过传递多个索引来编制索引。
# 对于高秩张量的每个单独的轴，遵循与单轴情形完全相同的索引规则。
print(rank_2_tensor.numpy())
# 为每个索引传递一个整数，结果是一个标量。
print(rank_2_tensor[1, 1].numpy())
# 您可以使用整数与切片的任意组合编制索引：
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

#%% 4 操作形状
# Reshaping a tensor is of great utility.
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
# You can convert this object into a Python list, too
print(var_x.shape.as_list())
# 通过重构可以改变张量的形状。重构的速度很快，资源消耗很低，因为不需要复制底层数据。
reshaped = tf.reshape(var_x,[1,3])
print(var_x.shape)
print(reshaped.shape)
# 数据在内存中的布局保持不变，同时使用请求的形状创建一个指向同一数据的新张量。
# TensorFlow 采用 C 样式的“行优先”内存访问顺序，即最右侧的索引值递增对应于内存中的单步位移。
# 如果您展平张量，则可以看到它在内存中的排列顺序。
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))

# 一般来说，tf.reshape 唯一合理的用途是用于合并或拆分相邻轴（或添加/移除 1）。
# 对于 3x2x5 张量，重构为 (3x2)x5 或 3x(2x5) 都合理，因为切片不会混淆：
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))

# 重构可以处理总元素个数相同的任何新形状，但是如果不遵从轴的顺序，则不会发挥任何作用。
# 利用 tf.reshape 无法实现轴的交换，要交换轴，您需要使用 tf.transpose。
# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")
# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")
# tf.transpose
print(tf.transpose(rank_3_tensor), "\n")

# 您可能会遇到非完全指定的形状。要么是形状包含 None 维度（维度的长度未知），要么是形状为 None（张量的秩未知）。
# 除了 tf.RaggedTensor 外，这种情况只会在 TensorFlow 的符号化计算图构建 API 环境中出现：tf.function/keras

#%% 5 Dtypes详解
# 使用 Tensor.dtype 属性可以检查 tf.Tensor 的数据类型。
# 从 Python 对象创建 tf.Tensor 时，您可以选择指定数据类型。

# 如果不指定，TensorFlow 会选择一个可以表示您的数据的数据类型。TensorFlow 将 Python 整数转换为 tf.int32，将 Python 浮点数转换为 tf.float32。
# 另外，当转换为数组时，TensorFlow 会采用与 NumPy 相同的规则。
# 使用cast将数据类型转换
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, let's cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)

#%% 6 广播
# 广播是从 NumPy 中的等效功能借用的一个概念。简而言之，在一定条件下，对一组张量执行组合运算时，为了适应大张量，会对小张量进行“扩展”。
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
# 同样，可以扩展大小为 1 的维度，使其符合其他参数。在同一个计算中可以同时扩展两个参数。
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
# 和矩阵运算一样，行乘列
print(tf.multiply(x, y))
# 下面是不使用广播的同一运算：
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading
# 在大多数情况下，广播的时间和空间效率更高，因为广播运算不会在内存中具体化扩展的张量。
# 使用 tf.broadcast_to 可以了解广播的运算方式。
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

#%% 7 tf.convert_to_tensor 转化为张量
# 大部分运算（如 tf.matmul 和 tf.reshape）会使用 tf.Tensor 类的参数。不过，在上面的示例中，您会发现我们经常传递形状类似于张量的 Python 对象
# 大部分（但并非全部）运算会在非张量参数上调用 convert_to_tensor。我们提供了一个转换注册表，大多数对象类
# （如 NumPy 的 ndarray、TensorShape、Python 列表和 tf.Variable）都可以自动转换。
# 有关更多详细信息，请参阅 tf.register_tensor_conversion_function。如果您有自己的类型，则可能希望自动转换为张量。 # 即支持第三方类型转换为张量


#%% 8 不规则张量
# 如果张量的某个轴上的元素个数可变，则称为“不规则”张量。对于不规则数据，请使用 tf.ragged.RaggedTensor。
# 例如，下面的例子无法用规则张量表示：
"""
0 1 2 3
4 5 
6 7 8
9
"""
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# 应使用 tf.ragged.constant 来创建 tf.RaggedTensor：
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
# tf.RaggedTensor 的形状包含未知维度：
print(ragged_tensor.shape)

#%% 9 字符串张量
# tf.string 是一种 dtype，也就是说，在张量中，我们可以用字符串（可变长度字节数组）来表示数据。
# 字符串是原子类型，无法像 Python 字符串一样编制索引。字符串的长度并不是张量的一个维度。有关操作字符串的函数，请参阅 tf.strings。
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
print(scalar_string_tensor.shape)
# 如果传递 Unicode 字符，则会使用 utf-8 编码。
tf.constant("🥳👍")
# 在 tf.strings 中可以找到用于操作字符串的一些基本函数，包括 tf.strings.split。
print(tf.strings.split(scalar_string_tensor, sep=" "))

# ...but it turns into a `RaggedTensor` if we split up a tensor of strings,
# as each string might be split into a different number of parts.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print(tensor_of_strings)
print(tf.strings.split(tensor_of_strings))

# tf.string.to_number：
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
# 虽然不能使用 tf.cast 将字符串张量转换为数值，但是可以先将其转换为字节，然后转换为数值。
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)

# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)

# tf.io 模块包含在数据与字节类型之间进行相互转换的函数，包括解码图像和解析 csv 的函数


#%% 10 稀疏张量
# 在某些情况下，数据很稀疏，比如说在一个非常宽的嵌入空间中。为了高效存储稀疏数据，TensorFlow 支持 tf.sparse.SparseTensor 和相关运算。
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# We can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
