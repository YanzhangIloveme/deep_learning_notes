#%% 以逻辑回归为例子引入

"""
Logistic Regression: sigmod(w^T*X+B), where sigmod(Z) = 1/(1+exp(-Z))
Cost Function: L(y_^, y)= - (y * logy_^ + (1-y) *log(1-y_^))
if   y=1, L = -logy_^, 为了让损失函数最小，所以max(y_^); 由于y_^=sigmod(Z) 属于(0,1),所以优化w使Zmax
elif y=0, L = -log(1-y_^), 为了是损失函数最小，所以最y_^ min-> 1-y_^max ->Lmin

损失函数是单训练样本定义的，定义成本函数衡量整个全体训练集表现
J(w,b) = 1/m * sumi=1,mL(y_^,y) = -1/m * sum([y(i)*y_^(i)+(1-y(i))*(1-y(i)_^)
所以训练过程是使J(w,b) min即可

"""

#%% Gradient Descent method

"""
一次迭代：
w = w - a * (d(J(w,b)/dw)
b = b - a * (d(J(w,b)/db)

"""

# 导数定义
# f'(x0) = limDelta(X)->0: [f(x+Delta(x)) - f(x0)] / Delta(X)

# 使用计算图法计算逻辑回归导数

# L(y_^, y)= - (y * logy_^ + (1-y) *log(1-y_^))
# 令a=y_^

# Z = wx+b -> a=sigmod(Z) => L(a,y)
# dL(a,y)/d(a) = -y/a + (1-y)/(1-a)
# d(a)/d(Z) = a(1-a)
# dL(a,y)/d(Z) = a(1-a) * (-y/a + (1-y)/(1-a)) = a-y
# d(Z)/d(w) = x
# d(Z)/d(b) = 1
# dL(a,y)/d(w) = x * (a-y)
# dL(a,y)/d(b) = (a-y)

# 适用于损失函数时，共m个组合
# J(w,b) = 1/m * sumi=1,mL(y_^,y) = -1/m * sum([y(i)*y_^(i)+(1-y(i))*(1-y(i)_^)

# dJ(w,b)/dw = 1/m * sum(dL(a,y)/d(w)) = 1/m * sum[ X(i) * (a(i)-y(i))]
# dJ(w,b)/db =  1/m * sum(dL(a,y)/d(b)) = 1/m * sum[a(i)-y(i)]
# 所以迭代累加后求平均即可
# 所以，把X，a(i)， y(i) 向量化求积，避免用for loop将会很高效。

#%% 向量化
import numpy as np
a = np.random.rand(100000)
b = np.random.rand(100000)
print(np.dot(a,b))
# 向量化后，使用dot两两相乘速度远远大于for loop

# case 2

# v = [v1, v2, v3] - > u = [exp(v1),...]
v = np.array([1,2,3])
u = np.exp(v)

# case 3 改写逻辑回归
# J = 0, dw1 =0, dw2=0, db = 0, m=1000
# for i in range (1,m):
#     z[i] = w^T*x(i) + b
#     a[i] = sigmod(Z[i])
#     j += -[y[i]*log(y_[i])+(1-a[i])*log(1-y_[i])]
#     dz[i] = a[i]-y[i]
#     dw1 += x1[i]*dz[i]  --> dw = np.zeros(n_x,1);dw+= x(i)dz(i)
#     dw2 += x2[i]*dz[i]
#     db += dz[i]
# j = j/m; dw1 = dw1/m; dw2=dw2/m, db = db/m

# -----------------------------------------------
# 全改写：
# z = np.dot(w.T,x)+b (b是实数 【1，1】， python会广播到所有到wT*x
# A = 1/exp(-Z)
# dZ = A-Y
# db = 1/m * np.sum(dZ)
# dw = 1/m * X * dz.T
# b -= alpha * db
# w -= alpha * dw

#%% 广播
# (m,n) =-*/ (1,n) -> (m,n)
# (m,n) =-*/ (m,1) -> (m,n)

#%% 逻辑回归证明：
"""
set yhat = p(y=1|x)
if y =1 : p(y|x) = yhat
if y =0 : p(y|x) = 1-yhat
so: p(y|x) =  yhat^y * (1-yhat)^(1-y)


so: MLE: maximum p(y|x) = > maximum log(p(y|x)) => y*logyhat + (1-y)*log(1-yhat) =>min -[y*logyhat + (1-y)*log(1-yhat)]


log(p(labels in target set)) => max log[sum(p(yhat|x)]

"""
