# 自动驾驶

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from yolo.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

import yolo.yolo_utils as yolo_utils

#%% CASE 说明：
# 假设你现在在做自动驾驶的汽车，你想着首先应该做一个汽车检测系统，为了搜集数据，你已经在你的汽车前引擎盖上安装了一个照相机，在你开车的时候它会每隔几秒拍摄一次前方的道路。
# y = (pc,bx,by,bh,bw,c)
# 假如你想让YOLO识别80个分类，你可以把分类标签c从1到80进行标记，或者把它变为80维的向量（80个数字），在对应位置填写上0或1。视频中我们使用的是后面的方案。
# 因为YOLO的模型训练起来是比较久的，我们将使用预先训练好的权重来进行使用。

"""
  YOLO（“you only look once”）因为它的实时高准确率，这就使得它是目前比较流行的算法。在算法中“只看一次（only looks once）”的机制使得它在预测时
  只需要进行一次前向传播，在使用非最大值抑制后，它与边界框一起输出识别对象。
  
  模型细节：
    输入对图片维度为(m, 608, 608, 3)
    输出为识别分类和边界框列表，由6个数字组成(pc,bx,by,bh,bw,c)；如果把c放到80维对向量中，则每个边界框就是85个数字组成；
    我们使用5个锚框： 算法流程: images -> DEEP CNN -> 编码(m, 19, 19, 5, 85) (5个锚框)
    如果对象的中心/中点在单元格内，那么该单元格就负责识别该对象
    为了方便，我们将把最后的两个维度的数据进行展开，所以最后一步的编码由(m,19,19,5,85)变为了(m,19,19,425)。

    对于每个单元格的每个锚框而言，我们将计算下列元素的乘积，并提取该框包含某一类的概率。
    scores = Pc * (c1,c2...c80) of box 1 -> find the max-> score:0.44, box (bx, by, bh, bw), class: c=3
    
    模型猜测的高概率的锚框，但锚框依旧是太多了。我们希望将算法的输出过滤为检测到的对象数量更少，要做到这一点，我们将使用非最大抑制。具体来说，我们将执行以下步骤：
    舍弃掉低概率的锚框（意思是格子算出来的概率比较低我们就不要）
    当几个锚框相互重叠并检测同一个物体时，只选择一个锚框。
    
    分类阈值过滤: 我们要为阈值进行过滤，去掉一些预测值低于预期值对锚框，模型共有19*19*5*85个数字，每个锚框85个数字，将其转化为：
    box_confidence：tensor类型，维度(19*19,5,1),包含19*19但愿个预测中对5个锚框所有的锚框pc
    boxes: tensor类型，（19*19，5，4）包含所有锚框的(px, py, ph,pw)
    box_class_probs： tensor类型，(19*19,5,80)，包含所有对象检测的概率
    
    我们要实现yolo_filter_boxes(),步骤：
    1 计算对象可能性
    2 对每个锚框，找到预测概率最大值对锚框（索引）
    3 根据阈值创建掩码，比如执行[0.9,0.3,0.4,0.5,0.1]<0.4,返回[f,t,f,f,t],对应我们要保留的锚框，其掩码为1/True
    4 使用tensorflow对box_class_scores\boxes\box_classes进行掩码操作过滤出我们的锚框

"""

#%% 锚框过滤器

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    通过阈值来过滤对象和分类的置信度。

    参数：
        box_confidence  - tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
        boxes - tensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。
        box_class_probs - tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。
        threshold - 实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。

    返回：
        scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。
        boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)
        classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引

    注意："None"是因为你不知道所选框的确切数量，因为它取决于阈值。
          比如：如果有10个锚框，scores的实际输出大小将是（10,）
    """
    # step 1 :calculate anchor box scores
    box_scores = box_confidence * box_class_probs
    # step 2 :find the index of anchor box with maximum score
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.keras.backend.max(box_scores, axis=-1)
    # step 3: find t/f threshold
    filtering_mask = (box_class_scores >= threshold)
    # step 4: find reminding scores\boxes\classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


# test
box_confidence = tf.random.normal([19,19,5,1], mean=1, stddev=4, seed=1)
boxes = tf.random.normal([19,19,5,4],  mean=1, stddev=4, seed=1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))
#%% 非最大值抑制

# 实现iou函数
def iou(box1, box2):
    """
    实现两个锚框的交并比的计算

    参数：
        box1 - 第一个锚框，元组类型，(x1, y1, x2, y2)
        box2 - 第二个锚框，元组类型，(x1, y1, x2, y2)

    返回：
        iou - 实数，交并比。
    """
    #计算相交的区域的面积
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1-xi2) * (yi1-yi2)
    #计算并集，公式为：Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area+box2_area-inter_area
    return inter_area/union_area

box1 = (2,1,4,3)
box2 = (1,2,3,4)

print("iou = " + str(iou(box1, box2)))

#%% 2 现在我们要实现非最大值抑制函数，关键步骤如下：
# 选择分值最高的锚框
# 计算与其他框重叠部分，删除与iou——threshold相比重叠的框
# 返回第一步直到没有框

# TensorFlow有两个内置函数用于实现非最大抑制（所以你实际上不需要使用你的iou()实现）：
# tf.image.non_max_suppression()
# K.gather()

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    为锚框实现非最大值抑制（ Non-max suppression (NMS)）

    参数：
        scores - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        boxes - tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小（见下文）
        classes - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        max_boxes - 整数，预测的锚框数量的最大值
        iou_threshold - 实数，交并比阈值。

    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类

    注意："None"是明显小于max_boxes的，这个函数也会改变scores、boxes、classes的维度，这会为下一步操作提供方便。

    """
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes, scores,max_boxes,iou_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


scores = tf.random.normal([54, ], mean=1, stddev=4, seed=1)
boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed=1)
classes = tf.random.normal([54, ], mean=1, stddev=4, seed=1)
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

#%% 对所有框进行过滤
def yolo_eval(yolo_outputs, image_shape=(720., 1280.),
              max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

    参数：
        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：
                        box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)
                        box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)
        image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)
        max_boxes - 整数，预测的锚框数量的最大值
        score_threshold - 实数，可能性阈值。
        iou_threshold - 实数，交并比阈值。

    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类
    """

    # 获取YOLO模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 可信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # 缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # 使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

yolo_outputs = (tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
scores, boxes, classes = yolo_eval(yolo_outputs)

print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))


#%% 对yolo总结
"""
输入图像为(608,608,3)

输入的图像先要通过一个CNN模型，返回一个(19,19,5,85)的数据。

在对最后两维降维之后，输出的维度变为了(19,19,425):
    每个19x19的单元格拥有425个数字。
    425 = 5 x 85，即每个单元格拥有5个锚框，每个锚框由5个基本信息+80个分类预测构成，参见图4。
    85 = 5 + 85，其中5个基本信息是 （pc, px, py, ph, pw),剩下80就是80个分类的预测。
    
然后我们会根据以下规则选择锚框：
    预测分数阈值：丢弃分数低于阈值的分类的锚框。
    非最大值抑制：计算交并比，并避免选择重叠框。
    
最后给出YOLO的最终输出。    
"""




#%% 测试已经训练好对yolo

class_names = yolo_utils.read_classes("yolo/models/model_data/coco_classes.txt")
anchors = yolo_utils.read_anchors("yolo/models/model_data/yolo_anchors.txt")
image_shape = (720.,1280.)
yolo_model = tf.keras.models.load_model("yolo/models/model_data/yolov2.h5")
yolo_model.summary()
# 将模型转换为边界框
# yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
# # 过滤锚框
# scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
# 在实际图像中运行计算图

#%%
def predict(image_file, is_show_info=True, is_plot=True):
    """
    运行存储在sess的计算图以预测image_file的边界框，打印出预测的图与信息。

    参数：
        sess - 包含了YOLO计算图的TensorFlow/Keras的会话。
        image_file - 存储在images文件夹下的图片名称
    返回：
        out_scores - tensor类型，维度为(None,)，锚框的预测的可能值。
        out_boxes - tensor类型，维度为(None,4)，包含了锚框位置信息。
        out_classes - tensor类型，维度为(None,)，锚框的预测的分类索引。
    """
    # 图像预处理
    image, image_data = yolo_utils.preprocess_image("yolo/images/" + image_file, model_image_size=(608, 608))

    # 运行会话并在feed_dict中选择正确的占位符.
    # 将模型转换为边界框
    prediction = yolo_model.predict(image_data).reshape(19,19,5,85)
    prediction_ls = (tf.Variable(prediction[:,:,:,0].reshape(19,19,5,1)),
                     tf.Variable(prediction[:,:,:,1:3].reshape(19,19,5,2)),
               tf.Variable(prediction[:, :, :, 3:5].reshape(19, 19, 5, 2)),
               tf.Variable(prediction[:, :, :, 5:].reshape(19, 19, 5, 80)))



    # 过滤锚框
    out_scores, out_boxes, out_classes = yolo_eval(prediction_ls, image_shape)

    # 打印预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(out_boxes.shape[1]) + "个锚框。")

    # 指定要绘制的边界框的颜色
    colors = yolo_utils.generate_colors(class_names)

    # 在图中绘制边界框
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存已经绘制了边界框的图

    image.save(image_file+'out.jpg', quality=100)

    # 打印出已经绘制了边界框的图
    if is_plot:
        output_image = plt.imread(image_file+'out.jpg')
        plt.imshow(output_image)
        plt.show()

    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict("0004.jpg")
