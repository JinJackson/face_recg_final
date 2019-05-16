import cv2
import glob
import os

face_dir = 'F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\output'
out_dir = 'F:\\Study\\FinalDesign\\Demo\\Wuenda\\face_recg\\gray_face'

for root, dirs, files in os.walk(face_dir):
    for file in files:
        img_path=root+'\\'+file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        cv2.imwrite(out_dir+'\\'+file, img)