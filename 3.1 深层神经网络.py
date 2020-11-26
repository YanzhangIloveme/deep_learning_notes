#%% 深度神经网络

# 多层的时候，其维度为：
# W[l] = (n[l], n[l-1])  # 右侧矩阵乘上上一层的输出，所以右侧为上一层的行数
# b[l] = (n[l), 1)
# dw[l] = (n[l], n[l-1])
# db[l] = (n[l], 1)

# Z[1] = W[1] * X + b[1] -> (n[1], m)
# Z[l], A[l] = (n[l], m)
# dZ[l], dA[l] = (n[l], m)

#%% 深层网络前向传播

# layer l: W[l], b[l]
# forward input : a[l-1], output: a[l]
# Z[l] = W[l]*a[l-1]+b[l]
# a[l] = g[l](Z[l])

# backward:
# input da[l], output da[l-1]
# cache Z[l], dW[l], db[l]


