import os
import numpy as np
import cv2

os.chdir(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv")
from sudoku_solver.solveSudoku2 import * 
from OCR.miscellaneous_sudoku import *
import image_processing.imageProc as imageProc
dir_name = ("C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv\sudoku_images\\test5")
filenames = os.listdir(dir_name)
Xtrain = []
ytrain = []
os.chdir(dir_name)
for file_name in filenames:
    #path = os.path.join(dir_name, file_name)
    dst,dst2  = imageProc.imagProc(file_name)
    i = 0
    j = 0
    idx_i = 0
    idx_j = 0
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx
    endy = 8*leny
    cut_off = blank_square_cut_off(dst)
    count = 0
    print "cut-off is " + str(cut_off)
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            img = img[4:-3,3:-4]# Changeable  4:-4,5:-5
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            #img = cv2.erode(img, kernel, iterations = 1)
            r = img.shape[0]
            c = img.shape[1]
            img2 = img.copy()
            print img.sum()
            if img.sum() < cut_off:               # Changeable  14000 
                cv2.imshow("sud",img)
                print "not taken in account"
            else:
                _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                max = 0
                max_idx = 0
                for k in range(len(contours)):
                    cnt = contours[k]
                    if cv2.contourArea(cnt) > max:
                        max = cv2.contourArea(cnt)
                        max_cnt = cnt
                cnt = max_cnt
                #epsilon = 0.01*cv2.arcLength(cnt,True)
                #approx = cv2.approxPolyDP(cnt,epsilon,True)
                
                #print cnt
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(img2,(x,y),(x+w,y+h),(255,255,255),1)
                #img2 = img2.resize((img.shape[0]+6,img.shape[1] + 6))
                cv2.imshow("dst",dst)
                img = img[y:y+h,x:x+w]
                cv2.imshow("sud",img)
                img = cv2.resize(img,(10,10))
                b = img.ravel()
                Xtrain.append(b)
                num = cv2.waitKey(0)
                ytrain.append(num-48)
            i+=lenx
            
            count += 1
            print idx_j, idx_i
            idx_i += 1
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        
        j+=leny
        idx_j+=1
    print Xtrain
    print ytrain
    
res = np.array(ytrain)
X = np.array(Xtrain)
res = res.reshape((len(res),1))
os.chdir(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv")
np.savetxt("feature6.txt",X)
np.savetxt("output6.txt",res)

