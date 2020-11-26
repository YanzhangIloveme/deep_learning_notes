#%% 1 目标定位：
# 定位分类问题：不仅要进行图片分类，还需要在图片上把目标使用框框标记出来其位置
# 对象检测：出现多个目标，如何识别多个目标及位置；

# 目标定位的大致思想：
# step1 使用softmax输出层获得多分类（pedestrain\car\motorcycle\background）
# step2 让神经网络多输出几个单元，输出一个边界框（多输出4个数字，标记为bx\by\bh\bw）
# 标记：图片左上角为（0，0） 右下角为（1，1）；其中图片中心点为bx,by,图片高bh,宽bw

# step3 为监督学习设置目标标签y = {pc, bx, by, bh, bw, c1, c2, c3} 其中pc是是否包含目标，c1..c3为是否是pedestrain\car\motorcycle分类
# 如果有目标： y = [1, bx, by, bh, bw, 0, 1, 0]
# 如果没目标： y = [0, ?, ?, ?, ?, ?, ?, ?]

# step4 定义损失函数: 其在y不同的时候损失函数并不相同
# L(y_, y) = (y1_ - y)**2 + ... + (yn_ - y) ** 2 if y ==1
#  L(y_, y) = (y1_ - y) **2

#%% 2 目标定位另一种思路：特征点landmark detection
# 其原理是通过对目标特征点的识别，可以让最后输出层多输出两个数字lx, ly
# 如人眼眶识别的时候，设置四个l1x, l1y, l2x, l2y, l3x, l3y ... 等分表表示其特征点；
# 可以让y输出为多个特征；通过训练模型可以检测特征点在哪里；但是这种训练数据标注非常费力；
# 可以通过训练标注的特征点来把我人的姿势；
# 注意的是，所有的标注数据特征点必须相应一直才可以；

#%% 3 基于滑动窗口的目标检测算法

# case： 以car detection为例：
# steps:
# 1 训练一个是否含有汽车的卷积神经网络
# 2 使用滑动窗口，逐步遍历图片中所有的小窗口，分别对其进行预测；
# 3 使用更大的region再次进行滑动窗口，进行预测；
# 4 则总有一个窗口可以检测到图片上的汽车；

# 其计算成本明显非常巨大；并且除非没有窗口期足够小，否则难以检测中心点；

#%% 4 在卷积层构建滑动窗口（大大降低计算成本）

# 1 turning FC layer into convolutional layers:
# 14*14*3 ->5*5 -> 10*10*16 -> max pool -> 5*5*16 -> FC（400） -> FC（400） -> y(softmax)
# 那么对FC（400）来说，如果使用5*5*16*400卷积，其实就是就变成了1*1*400；这样完成率fc到卷积对转化
# 再进行1*1*400卷积，则完成第二个FC(400)，在进行1*1->1*1*4

# 2 通过卷积化进行滑动窗口
# 我们不需要把输入图片分割成四个子集，分别前向传播；把他们当作一个图片，则可以共享很多计算；
# 如原来是14*14*3 ，补成16*16*3 -5*5 -> 12*12*16 -> max pool 2*2-> 6*6*16 ->
# 2*2*400 -> 1*1 -> 2*2*400 -> 1*1 -> 2*2*4 （则最后对2*2中，左上角对应原始图片左上角，位置依次对应）
# 这个算法提升了效率，但是边界框的位置仍然不稳定；

#%% 5 Bounding Box预测

# 引入：YOLO算法： you only look once
# 假设有100*100的交通照片，画上3*3网格（实际中存在更精细的网格）
# 对每个网格应用图像分类和定位算法；
"""
labels for trainning

for each grid cell:
    y = [pc, bx, by, bh, bw, c1, c2, c3]
    注：我们标注的时候标注中心点在哪个cell
所以总输出为3*3*8；

该算法的优势是可以较准确的得出对象的边界

"""

# 如何编码bx by？
"""
约定cell的左上角为(0,0),右下角为(1,1); 然后bx,by,bh,bw按比例生成即可；
如果该图片跨cell了，那么bh,bw可能大于1,但是非负数！
"""

#%% 6 交并比 -- 如何判断对象检测算法运作良好

# Intersection Over Union (IOU):
# 计算 size of intersection / size of union: "Correct: if IoU>=0.5"


#%% 7 非极大值抑制 Non-max suppression
# 该思想确保算法detects each object only once
# 如19*19 YOLO，共361个格子，这样一辆车可能多个格子都会告诉中心点可能在我这里。非极大抑制保证只检测一次；
"""
算法流程：
1. 比较多个格子的Pc，找最大的这个格子；
2. 逐一审视剩下的矩形，所有有这个最大的边界框很高的交并比，高度重叠的其他边界框，将他们的输出进行抑制！
3. 迭代遍历完所有的格子；
non-max表示：只输出概率最大的分类结果，然后抑制接近但非最大但结果


算法细节：

1. each output prediction [pc, bx, by, bh, bw, c1...], 共391个
2. 去掉pc小于等于某个阈值，如0.6的box
3. while remaining boxes:
    pick the box with the largest pc, output that as a prediction
    discard any remainning box with IoU>=0.5 with largest box
4. 没有剩下的box了，算法完成

（如果有多个c1,c2,c3，则需要对每个类别输出都做一次non-max suppression）
"""

#%% 8 Anchor Boxes

# 目前算法问题在于，每个格子只可以检测出一个对象；如果希望一个box检测出多个对象，则可以使用anchor boxes
# 如果有一张3*3 boxes， 一个人在汽车前面
# 则如果   y = [pc, bx, by, bh, bw, c1, c2, c3]，则无法检测多个；
# 引入anchor box，定义多个anchor box，把预测结果和这两个anchor box关联起来；
# 则 y = [pc, bx, by, bh, bw, c1, c2, c3, pc, bx, by, bh, bw, c1, c2, c3]
#               anchor box 1                          anchor box 2

# Reviously: 每一个网格归属目标的中心点
# anchor box：每一个目标都归属于含有目标中点和anchor box with highest IoU的cell
# 输出为(grid cell, anchor box)
# 但是，如果同一个cell含有3个对象，则该算法表现不好；
# 算法加入anchor boex可以更加有针对性的发现目标；

#%% 9 组合：YOLU算法

# CASE： y is 3*3*2*8
# c1: pedestrain c2: car c3：motorcycle
# 使用两个anchor box： # 则 y = [pc, bx, by, bh, bw, c1, c2, c3, pc, bx, by, bh, bw, c1, c2, c3]
# 则输出是3*3*2*8

"""
1 构建训练集： 遍历9个格子；生成相应的y；如左上角啥都没，他就是[0,???????,0???????];
                                  如果其中一个有car [0???????,1,bx,by,bh,bw,0,1,0]
    
2.训练，输入100*100*3 输出3*3*16
3.run non-max suppression：如两个anchor box，则9个格子每一个都会有两个预测的边界框； -> get rid of low pc -> 对每个类别单独
  运行non-max suppression
"""

#%% 10 RPN网络：Region proposals （后续区域）

# R-CNN网络：带区域的卷积网络，算法尝试选出一些区域；在这些区域运行卷积网络分类是有意义的
# 选候选区的方法是运行图像分割算法
# 但是R-cnn还是很慢

# Fast R-CNN (FR-CNN):使用滑动窗口的卷积实现提升了R-cnn速度
# 但是得到候选区域的聚类步骤仍然缓慢

# Faster R-CNN：更快！使用卷积神经网络而不是传统聚类算法，实现更快
# 但还是比YOLO更慢
