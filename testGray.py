#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####
from bokeh import model
from keras.backend import get_session
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras import models
import glob
#------------用于绘制模型细节，可选--------------#
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#------------------------------------------------#
K.set_image_data_format('channels_first')
import time
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import dlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from inception_blocks_v2 import *
import h5py
from fr_utils import *
#获取模型
size = 96
# FRmodel = faceRecoModel(input_shape=(3,size,size))



#打印模型的总参数数量
# print("参数数量：" + str(FRmodel.count_params()))

#加载所有人脸数据
print("初始化人脸数据字典")
dic = {}
file_dir = "F:/Study/FinalDesign/Demo/Wuenda/face_recg/face_vec_gray"
for root, dirs, files in os.walk(file_dir):
    for file in files:
        dic[os.path.splitext(file)[0]] = np.loadtxt("face_vec_gray/" + str(file))
print("初始化完成")


def triplet_loss(y_true, y_pred, alpha = 0.2):
    # 根据公式（4）实现三元组损失函数
    #
    # 参数：
    #     y_true -- true标签，当你在Keras里定义了一个损失函数的时候需要它，但是这里不需要。
    #     y_pred -- 列表类型，包含了如下参数：
    #         anchor -- 给定的“anchor”图像的编码，维度为(None,128)
    #         positive -- “positive”图像的编码，维度为(None,128)
    #         negative -- “negative”图像的编码，维度为(None,128)
    #     alpha -- 超参数，阈值
    #
    # 返回：
    #     loss -- 实数，损失的值
        #获取anchor, positive, negative的图像编码
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        #第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
        #第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
        #第三步：减去之前的两个距离，然后加上alpha
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
        #通过取带零的最大值和对训练样本的求和来计算整个公式
        loss = tf.reduce_sum(tf.maximum(basic_loss,0))
        return loss





print("--------start loading---------")
FRmodel = models.load_model("FRmodel.h5", custom_objects={'triplet_loss': triplet_loss})
print("--------end loading---------")



# save_New_gray("gray_face/jzl.jpg","jzl",FRmodel)
# save_New_gray("gray_face/wf.jpg","wf",FRmodel)
# save_New_gray("gray_face/zc.jpg","zc",FRmodel)
# save_New_gray("gray_face/wzh.jpg","wzh",FRmodel)
# save_New_gray("gray_face/zhb.jpg","zhb",FRmodel)
# save_New_gray("gray_face/jh.jpg","jh",FRmodel)
# save_New_gray("gray_face/jzlN.jpg","jzlN",FRmodel)




def verify(image_path, name, model):
    # 第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    tmp_path = 'F:/Study/FinalDesign/Demo/Wuenda/face_recg/tmp/tmp.jpg'
    for i, d in enumerate(dets):
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (size, size))
        # cv2.imshow('image', face)
        cv2.imwrite(tmp_path, face)
    print("start encoding")
    encoding = img_to_encoding(tmp_path, model)
    #第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - dic[name])
    #第三步：判断是否打开门
    if dist < 0.9:
        print("欢迎 " + str(name) + "回家！")
        print("distance"+str(dist))
        is_door_open = True
    else:
        print("distance" + str(dist))
        print("经验证，您与" + str(name) + "不符！")
        is_door_open = False
    return dist, is_door_open


def who_is_it(image_path, model):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)

    count = 0
    print("共有人脸" + str(len(dets)) + "个")
    for i, d in enumerate(dets):
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (size, size))
        # cv2.imshow('image', face)
        tmp_path = 'F:/Study/FinalDesign/Demo/Wuenda/face_recg/tmp/tmp' + str(count) + '.jpg'
        cv2.imwrite(tmp_path, face)
        encoding = img_to_encoding(tmp_path, model)
        min_dist = 100
        # print(model.summary()) 打印网络结构
        for key in dic:
            dist = np.linalg.norm(encoding - dic[key])
            if dist < min_dist:
                min_dist = dist
                identity = key
        if min_dist > 0.9:
            print("Not in the database.")
        else:
            print("It's " + str(identity) + ", the distance is " + str(min_dist))
        # return min_dist, identity
        # count = count + 1
    return identity

# who_is_it('tmp/tmp.jpg',FRmodel)
# verify('tmp/tmp.jpg',"jh",FRmodel)



testpic_dir = 'F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\test2'
result = []
correct = 0
count = 0
for im in glob.glob(testpic_dir + '\\*.jpg'):
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('tmp/tmp_gray.jpg', img)
    name = who_is_it('tmp/tmp_gray.jpg', FRmodel)
    print(name)
    count = count + 1
    if name == 'jzlN':
        correct = correct + 1
        print("accuracy："+str(correct / count))


# 使用摄像头识别
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', frame)
#     # 抓取图像
#     if cv2.waitKey(1) & 0xFF == ord('S'):
#         cv2.imwrite('tmp/tmp.jpg', frame)
#         break

# who_is_it('tmp/tmp.jpg',FRmodel)


