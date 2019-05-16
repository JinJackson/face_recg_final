import cv2
import dlib
import random
import os
from PIL import Image
import glob






#调整目标目录的jpg文件像素
def file_name(file_dir):
    count = 1
    for im in glob.glob(file_dir + '\\*.png'):
        img = cv2.imread(im)
        out = cv2.resize(img,(96,96))
        cv2.imwrite('F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\images2\\output\\' + str(count) + '.jpg',out)
        count += 1




#获取目标文件夹照片的脸部并裁剪
def get_Face(path):
    size = 128
    detector = dlib.get_frontal_face_detector()
    count = 1
    for im in glob.glob(path+'\\*.jpg'):
            img = cv2.imread(im)
            # print(img.shape)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            for i, d in enumerate(dets):
                    y1 = d.top() if d.top() > 0 else 0
                    y2 = d.bottom() if d.bottom() > 0 else 0
                    x1 = d.left() if d.left() > 0 else 0
                    x2 = d.right() if d.right() > 0 else 0
                    face = img[y1:y2, x1:x2]
                    face = cv2.resize(face, (size, size))
                    cv2.imshow('image', face)
                    cv2.imwrite('F:\\Study\\Final Design\\Workspace\\face_recog_jin\\cl2\\' + str(count) + '.jpg', face)
                    count += 1

def get_Face2(path):
    print("start")
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    size = 96
    detector = dlib.get_frontal_face_detector()
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'\\*.JPG'):
            img = cv2.imread(im)
            print(img.shape)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            for i, d in enumerate(dets):
                    y1 = d.top() if d.top() > 0 else 0
                    y2 = d.bottom() if d.bottom() > 0 else 0
                    x1 = d.left() if d.left() > 0 else 0
                    x2 = d.right() if d.right() > 0 else 0
                    face = img[y1:y2, x1:x2]
                    face = cv2.resize(face, (size, size))
                    # cv2.imshow('image', face)
                    cv2.imwrite('F:/Study/FinalDesign/Demo/Wuenda/face_recg/output/' + str(idx) + '.jpg', face)

def get_single_face(im,name):
    size = 96
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(im)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    for i, d in enumerate(dets):
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (size, size))
        # cv2.imshow('image', face)
        cv2.imwrite('F:/Study/FinalDesign/Demo/Wuenda/face_recg/output/' + name + '.jpg', face)

path = 'F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\images3\\wzh\\4.jpg'



