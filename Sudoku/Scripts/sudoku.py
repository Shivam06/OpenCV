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

