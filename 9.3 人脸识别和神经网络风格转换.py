#%% 1 人脸识别
# Verification:
# - Input image, name/ID
# - Output whether the input image is that of the claimed person

# Recognition:
# - Has a database of K persons
# - Get an input image
# - Output ID if the image is any of the K persons (or not recognized)

#%% 2 One-shot learning
# Learning from one example to recognize the person again
# 核心是similarity function： d(img1, img2) = degree of difference between images
# If d(img1, img2) <= tau  then "same"  else "different"

#%% 3 Siamese 网络
"""
x1: face1 -> nn network -> f(x1) (128)
x2: face2 -> nn network -> f(x2) (128)
d(x1,x2) =  ||f(x1)-f(x2)||**2
如果x1, x2 是同一个人，则d(x1,x2) 较小，反之较大
"""
"""
Siamese网络训练方法：triplet loss
损失存在两部分，第一个是anchor，第二个是positive case，第三个是negative case
Target: ||f(A) - f(p)|| - ||f(A) - f(n)|| + alpha <=0 where alpha 为大于0的小数
三元损失的核心是，和True案例的距离远远小于和False案例的距离。

L(A,P,N) = max(||f(A)-f(p)||^2 - ||f(A)-f(n)||^2, 0)
During training, if A,P,N are chosen randomly, d(A, P) + a <= d(A,N) is easily satisfied.
Choose triplets that're "hard" to train on.


"""

#%% 4 另外一种架构：
# 在Siamese网络最后一层加上一层sigmoid层输出0、1

#%% 5 神经网络风格转换

"""
J(G) = a * Jcontent(C,G) + b * Jtyle(S,G)
 - Initiate G randomly
 - Use gradient descent to min J(G)
 - G:= G - delta(J(G))

Content cost function: Jcontent(c,g) = 1/2 * ||a[l][c] - a[l][g]||**2
Use hidden layer l to compute content cost.
Use pre-trained ConvNet.
Let a[l][c] and a[l][g] be the activaation of layer l on the images
If a[l][c] and a[l][g] are similar, both images have similar content

Style cost function:
Say you are using layer l's activation to measure "style".
Define style as corr between activations across channels

"""


