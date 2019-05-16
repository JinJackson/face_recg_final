import face_recognition as fr
import numpy as np
import cv2
import dlib
from PIL import Image

img1 = "F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\images3\\jzl\\jzl.jpg"
img2 = "C:\\Users\\Jackson\\Desktop\\test2.jpg"
size = 96

def getEmbedding(img_path):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    tmp_path = 'F:/Study/FinalDesign/Demo/Wuenda/face_recg/tmp/tmp.jpg'
    if len(dets) >= 2:
        print(len(dets))
        print("检测到多张人脸，请重新运行")
        return np.array(0)
    else:
        for i, d in enumerate(dets):
            y1 = d.top() if d.top() > 0 else 0
            y2 = d.bottom() if d.bottom() > 0 else 0
            x1 = d.left() if d.left() > 0 else 0
            x2 = d.right() if d.right() > 0 else 0
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (size, size))
            embedding = fr.face_encodings(face)
        return embedding

#
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', frame)
#
#     # 抓取图像
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite('tmp/tmp.jpg', frame)
#         break

embedding1 = np.array(getEmbedding(img1))
embedding2 = np.array(getEmbedding(img2))

# im1 = cv2.imread(img1)
# gray_image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# im1 = cv2.resize(gray_image1,(size,size))
# embedding1 = np.array(fr.face_encodings(im1))
#
# im2 = cv2.imread(img2)
# gray_image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# im2 = cv2.resize(gray_image2,(size,size))
# embedding2 = np.array(fr.face_encodings(im2))


dist = np.linalg.norm(embedding1 - embedding2)
print(dist)





