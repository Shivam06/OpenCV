import cv2
import numpy as np
import os


img = cv2.imread("C:\Users\SHIVAM MAHAJAN\Desktop\sudoku.jpg",cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray,5,150,150)
a = 30*np.ones(img_gray.shape,np.uint8)
img_gray = cv2.add(img_gray, a)
img_gray = cv2.multiply(img_gray, 1.2)
#lap = cv2.Laplacian(img_gray,cv2.CV_64F,ksize = 5)
drawing = np.zeros(img.shape,np.uint8)
th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
_,contours,_ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_idx,max_area = 0, 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if (area > max_area):
        max_area = area
        max_idx = i
    else:
        continue

cnt = contours[max_idx]
#cv2.drawContours(img,[cnt],-1,(0,255,0),1)
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(img,[approx],-1,(255,0,0),1)
app = []
print approx
for i in range(4):
    print approx[i][0]
    app.append(approx[i][0])
pts1 = np.float32(app)
pts2 = np.float32([[300,0],[0,0],[0,300],[300,300]])
print app
print pts2
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
dst = dst[7:,:]
cv2.imshow("image",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
         
