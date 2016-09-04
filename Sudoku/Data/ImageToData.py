import numpy as np
import cv2
import os
import sys

os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop")
img = cv2.imread("numbers2.png",0)
img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
ret, th = cv2.threshold(img, 10,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3),np.uint8)
th = cv2.dilate(th, kernel, iterations = 1)
_,contours,_ = cv2.findContours(th, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
res = []
X = []
keys = [i for i in range(48,58)]
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 100 and cv2.arcLength(contours[i],True) > 70:
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("image",img2)
        mat = img[y:y+h,x:x+w]
        cv2.imshow("img",mat)
        mat = cv2.resize(mat,(10,10))
        num = cv2.waitKey(0)
        if num == 27:
            break
        elif num in keys:
            res.append(num-48)
            arr = mat.ravel()
            X.append(arr)

res = np.array(res)
X = np.array(X)
res = res.reshape((res.shape[0],1))
np.savetxt("inputs_train.data",X)
np.savetxt("response_train.txt",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
