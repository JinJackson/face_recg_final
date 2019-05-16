# get_my_faces_dlib.py
import cv2
import dlib
import random
import os
# import sys
from time import sleep
out_dir = 'F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\test4_highlight2'
size = 128
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def relight(img, light=1, beta=0):
    w = img.shape[1]
    h = img.shape[0]
    # h, w, _ = img.shape

    for i in range(h):
        for j in range(w):
            for c in range(3):
                temp = int(img[i,j,c]*light+beta)
                if temp>255:
                    temp=255
                elif temp<0:
                    temp=0
                img[i,j,c] = temp
    return img

cam = cv2.VideoCapture(0)
index = 1
while index <=240:
         # print('Being processed picture %s' % index)
         success, img = cam.read()
         print(success)
         print(img.shape)
         pic = cv2.resize(img, (size, size))
         cv2.imshow('image', img)
         sleep(0.05)
         cv2.imwrite(out_dir + '/' + str(index) + '.jpg', img)
         index += 1

#截取人脸拍照
# detector = dlib.get_frontal_face_detector()
# cam = cv2.VideoCapture(0)
#
# index = 1
# while True:
#     if index<=1:
#         print('Being processed picture %s' % index)
#         success, img = cam.read()
#         print(success)
#         print(img.shape)
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         dets = detector(gray_image, 1)  # 返回是一个框[(x1,y1),(x2,y2)]
#         for i, d in enumerate(dets):
#             y1 = d.top() if d.top()>0 else 0
#             y2 = d.bottom() if d.bottom()>0 else 0
#             x1 = d.left() if d.left()>0 else 0
#             x2 = d.right() if d.right()>0 else 0
#
#             face = img[y1:y2, x1:x2]
#             face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
#
#             face = cv2.resize(face, (size, size))
#             cv2.imshow('image', face)
#             cv2.imwrite(out_dir+'/'+str(index)+'.jpg',face)
#             index += 1
#         key = cv2.waitKey(30) & 0xff
#         if key==27:
#             break
#     else:
#         print('finished')
#         break
