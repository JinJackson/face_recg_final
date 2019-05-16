from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
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

from inception_blocks_v2 import *
import h5py
from fr_utils import *
import dlib
#获取模型
FRmodel = faceRecoModel(input_shape=(3,96,96))

#打印模型的总参数数量
print("参数数量：" + str(FRmodel.count_params()))

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

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config):
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
fr_utils.load_weights_from_FaceNet(FRmodel)


# save_New("images2/mom.jpg", "mom", FRmodel)
# save_New("images2/dad.jpg", "dad", FRmodel)
# save_New("images2/cl.jpg", "cl", FRmodel)
# save_New("images2/fbb.jpg", "fbb", FRmodel)
# save_New("images2/yaleB01.jpg", "yaleB01", FRmodel)
# save_New("images2/obama.jpg", "obama", FRmodel)
# save_New("images2/jzl.jpg", "jzl", FRmodel)
# save_New("images2/hy.jpg", "hy", FRmodel)

save_New("output/wzh.jpg","wzh",FRmodel)
save_New("output/zc.jpg","zc",FRmodel)
save_New("output/zhb.jpg","zhb",FRmodel)


# np.savetxt("face_vec/mom.txt", mom)
# np.savetxt("face_vec/dad.txt", dad)
# np.savetxt("face_vec/cl.txt", cl)
# np.savetxt("face_vec/fbb.txt", fbb)
# np.savetxt("face_vec/yale01.txt", yale01)
# np.savetxt("face_vec/obama.txt", obama)
# np.savetxt("face_vec/jzl.txt", jzl)
# np.savetxt("face_vec/hy.txt", hy)
#
# x = np.loadtxt("face_vec/mom.txt")
#
# print (x)

# def convertDic(name ,value, dic):
#     dic[name] = value
#
# convertDic("mom",mom,vector_list)
# convertDic("dad",dad,vector_list)
# convertDic("cl",cl,vector_list)
# convertDic("fbb",fbb,vector_list)
# convertDic("yale01",yale01,vector_list)
# convertDic("obama",obama,vector_list)
# convertDic("jzl",jzl,vector_list)
# convertDic("hy",hy,vector_list)












