import cv2
import numpy as np

import os
os.chdir(r"C:/Users/SHIVAM MAHAJAN/Desktop")

img = cv2.imread("bookpage.jpg",cv2.IMREAD_COLOR)
#retval, threshold = cv2.threshold(img,12, 255,cv2.THRESH_BINARY)
img[0:7,0:7] = [198,198,198]
#print img[400,400]
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,threshold2 = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
#print img2gray[400,400]
i = 3
font = cv2.FONT_HERSHEY_COMPLEX
while i <=27:
    th2 = cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,i,2)
    
    cv2.putText(th2,"Block Sixe is %d X %d "%(i,i),(100,100),font,1,(0,255,0),5)
    cv2.imshow("win1",th2)
    i+=2
    cv2.waitKey(0)
    
#gauss = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,0)
cv2.imshow("orig",img2gray)


cv2.destroyAllWindows()
