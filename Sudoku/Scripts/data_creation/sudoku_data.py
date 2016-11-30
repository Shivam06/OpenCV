import numpy as np
import cv2
import os
import sys

os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\OpenCV Images")
img = cv2.imread("numbers2.png",0)
img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
ret, th = cv2.threshold(img, 100,255,cv2.THRESH_BINARY)
cv2.imshow("image1",th)
th = cv2.bilateralFilter(th,5,80,80)
kernel = np.ones((3,3),np.uint8)


th = cv2.bitwise_not(th)
#th = cv2.erode(th, kernel, iterations = 1)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
#cv2.imshow("image",th)

cv2.imshow("imag",th)
img3 = th.copy()
_,contours,_ = cv2.findContours(img3, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img2,contours,-1,(0,255,0),1)
#cv2.imshow("image2",img2)
cv2.imshow("image1",th)
res = []
X = []
keys = [i for i in range(48,58)]
count = 0
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 120 and cv2.arcLength(contours[i],True) > 70:
        count += 1
        
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(th,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("image2",th)
        mat = th[y:y+h,x:x+w]
        
        mat = cv2.resize(mat,(10,10))
        cv2.imshow("img",mat)
        num = cv2.waitKey(0)
        cv2.destroyWindow("img")
        cv2.destroyWindow("image2")
        if num == 27:
            break
        elif num in keys:
            res.append(num-48)
            arr = mat.ravel()
            X.append(arr)
            
        

print count

res = np.array(res)
X = np.array(X)
res = res.reshape((res.shape[0],1))
print X
os.chdir(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv")
np.savetxt("trial1.txt",X)
np.savetxt("trial2.txt",res)


cv2.waitKey(0)
cv2.destroyAllWindows()
