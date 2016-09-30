import cv2
import numpy as np
import os
from kNearestNeighbour import KNearestNeighbour
img = cv2.imread("C:\Users\SHIVAM MAHAJAN\Desktop\suduku3.png",cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, img_gray = cv2.threshold(img_gray,220,255,cv2.THRESH_BINARY_INV) #--
cv2.imshow("image3",img_gray)
os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop")  
Xtrain = np.loadtxt("inputs_train.data")
ytrain = np.loadtxt("response_train.txt")      #--

"""img_gray = cv2.bilateralFilter(img_gray,5,150,150)
a = 30*np.ones(img_gray.shape,np.uint8)
img_gray = cv2.add(img_gray, a)
img_gray = cv2.multiply(img_gray, 1.2)
cv2.imshow("hel",img_gray)
#lap = cv2.Laplacian(img_gray,cv2.CV_64F,ksize = 5)
drawing = np.zeros(img.shape,np.uint8)
th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
cv2.imshow("tt",th)
img3 = th.copy()
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
#print approx
for i in range(4):
    #print approx[i][0]
    app.append(approx[i][0])
pts1 = np.float32(app)
pts2 = np.float32([[300,0],[0,0],[0,300],[300,300]])
#print app
#print pts2
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img3,M,(300,300))
dst = dst[8:,8:]

#ret, dst = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
cv2.imshow("image",dst)
print dst.shape
#dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

"""


dst = img_gray              #--
i = 0
j = 0
lenx = dst.shape[1]/9.0
leny = dst.shape[0]/9.0
endx = 8*lenx
endy = 8*leny
arr = np.zeros((9,9))
knn = KNearestNeighbour(1) 
knn.fit(Xtrain,ytrain)      # --
idx_j = 0
a = []
while j <= endy:
    i = 0
    idx_i = 0
    while i <=endx:
        img = dst[j:j+leny,i:i+lenx]
        img = img[4:-4,4:-4]                # Changeable
        img2 = img.copy()
        print img.sum()
        if img.sum() < 10000:
            arr[idx_j,idx_i] = 0
            
        else:
            _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max = 0
            max_idx = 0
            for k in range(len(contours)):
                cnt = contours[k]
                if cv2.contourArea(cnt) > max:
                    max = cv2.contourArea(cnt)
                    max_cnt = cnt
            cnt = contours[max_idx]
            #print cnt
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,255,255),1)
            img = img[y:y+h,x:x+w]
            cv2.imshow("sud",img)
            img = cv2.resize(img,(10,10))
            b = img.ravel()
            arr[idx_j,idx_i] = knn.predict(b)[0]
            print arr[idx_j,idx_i]
        i+=lenx
        idx_i+=1
        cv2.waitKey(2000)
        cv2.destroyWindow("sud")
    j+=leny
    idx_j+=1
cv2.waitKey(0)
cv2.destroyAllWindows()


        
print arr

