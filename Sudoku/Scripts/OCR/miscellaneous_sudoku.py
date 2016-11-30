import cv2
from kNearestNeighbour import KNearestNeighbour
import numpy as np
import os
from neuralNet import Network


def show_solved_sudoku(dst,dst2,arr,thick=0):
    #cv2.imshow("dst",dst)
    #cv2.imshow("dst2",dst2)
    cv2.waitKey(0)
    print "array is"
    print arr
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx + lenx/5.0
    endy = 8*leny + leny/5.0
    idx_j = 0
    i = 0
    j = 0
    cut_off = blank_square_cut_off(dst,thick)
    print "cutoff is " + str(cut_off)
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            if idx_j%3 == 0 and idx_i%3 == 0:
                img = img[4+thick:-4-thick,4+thick:-4-thick]
            elif idx_i%3 == 0:
                img = img[4+thick:-3-thick,4+thick:-4-thick]# Changeable  4:-4,5:-5
            elif idx_j%3 == 2:
                img = img[4+thick:-4-thick,3+thick:-4-thick]
            else: 
                img = img[4+thick:-3-thick,3+thick:-4-thick]
            print img.sum()
            if img.sum() < cut_off:
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(dst2,str(int(arr[idx_j,idx_i])),(int(i+lenx/2.5),int(j+leny/1.5)),font,0.75,(0,255,0),2)  # 0.64
                ##cv2.imshow("imgg",img)
                #dst2[j:j+leny,i:i+lenx] = img.resize((dst2.shape[0],dst2.shape[1]))
                #cv2.waitKey(0)
            i+=lenx
            idx_i+=1
        j+=leny
        idx_j+=1



    cv2.imshow("final_output",dst2)
    cv2.imwrite("final.jpg",dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocr_knn(dst,dst2,neighbors=5,thick=0):
    cv2.imshow("OCR",dst2)
    Xtrain = np.loadtxt("final_feature.txt")
    ytrain = np.loadtxt("final_output.txt")
    i = 0
    j = 0
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx + lenx/5.0
    endy = 8*leny + leny/5.0
    arr = np.zeros((9,9))
    knn = KNearestNeighbour(neighbors) 
    knn.fit(Xtrain,ytrain)      # --
    cut_off = blank_square_cut_off(dst,thick)
    print "cut-off is " + str(cut_off)
    idx_j = 0
    a = []
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            if idx_j%3 == 0 and idx_i%3 == 0:
                img = img[4+thick:-4-thick,4+thick:-4-thick]
            elif idx_i%3 == 0:
                img = img[4+thick:-3-thick,4+thick:-4-thick]# Changeable  4:-4,5:-5
            elif idx_j%3 == 2:
                img = img[4+thick:-4-thick,3+thick:-4-thick]
            else: 
                img = img[4+thick:-3-thick,3+thick:-4-thick]
            r = img.shape[0]
            c = img.shape[1]
            img2 = img.copy()
            print img.sum()
            cv2.imshow("sud",img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            if img.sum() < cut_off:               
                arr[idx_j,idx_i] = int(0)
            else:
                _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                max = 0
                max_idx = 0
                #print "Lenth of contour is" + str(len(contours))
                for k in range(len(contours)):
                    cnt = contours[k]
                    if cv2.contourArea(cnt) > max:
                        max = cv2.contourArea(cnt)
                        max_idx = k
                cnt = contours[max_idx]
                #print cnt
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(dst2,(int(i),int(j)),(int(i+lenx),int(j+leny)),(0,0,255),3)
                #print img.shape
                img = img[y:y+h,x:x+w]
                img = cv2.resize(img,(10,10))
                cv2.imshow("sud",img)
                b = img.ravel()
                arr[idx_j,idx_i] = int(knn.predict(b))
                print "Recognized number is " + str(arr[idx_j,idx_i])
                print "Percent Score is " + str(knn.percentage_score()) + "\n"
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(dst2,str(int(arr[idx_j,idx_i])),(int(i),int(j+leny*0.9)),font,0.75,(0,255,0),2)
                cv2.imshow("OCR",dst2)
                cv2.imwrite("segmented.jpg",dst2)
                cv2.waitKey(1000)
            i+=lenx
            idx_i+=1
            cv2.destroyWindow("sudo")
        j+=leny
        idx_j+=1
    print arr
    cv2.destroyAllWindows()
    cv2.imwrite("ocr.jpg",dst2)
    return arr


def ocr_nn(dst,dst2,thick = 1):
    i = 0
    j = 0
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx + lenx/5.0
    endy = 8*leny + leny/5.0
    arr = np.zeros((9,9))
    net = Network([784,30,10])
    net.get_weights()
    net.get_biases()     
    cut_off = blank_square_cut_off(dst,thick)
    print "cut-off is " + str(cut_off)
    idx_j = 0
    a = []
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            if idx_j%3 == 0 and idx_i%3 == 0:
                img = img[4+thick:-4-thick,4+thick:-4-thick]
            elif idx_i%3 == 0:
                img = img[4+thick:-3-thick,4+thick:-4-thick]
            elif idx_j%3 == 2:
                img = img[4+thick:-4-thick,3+thick:-4-thick]
            else: 
                img = img[4+thick:-3-thick,3+thick:-4-thick]
            r = img.shape[0]
            c = img.shape[1]
            img2 = img.copy()
            print img.sum()
            cv2.imshow("sud",img)
            if img.sum() < cut_off:             \
                arr[idx_j,idx_i] = int(0)  
            else:
                _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                max = 0
                max_idx = 0
                #print "Lenth of contour is" + str(len(contours))
                for k in range(len(contours)):
                    cnt = contours[k]
                    if cv2.contourArea(cnt) > max:
                        max = cv2.contourArea(cnt)
                        max_idx = k
                cnt = contours[max_idx]
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(dst2,(int(i),int(j)),(int(i+lenx),int(j+leny)),(0,0,255),3)
                #print img.shape
                img = img[y:y+h,x:x+w]
                img = cv2.resize(img,(12,16))
                img = np.array(img)
                print img.shape
                img = np.vstack((img,np.zeros((6,12))))
                img = np.vstack((np.zeros((6,12)),img))
                img = np.hstack((img,np.zeros((28,8))))
                img = np.hstack((np.zeros((28,8)),img))
                cv2.imshow("sud",img)
                b = img.ravel()
                b = b/255.0
                b = b.reshape((784,1))
                arr[idx_j,idx_i] = int(net.predict(b))
                print "Recognized number is " + str(arr[idx_j,idx_i])
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(dst2,str(int(arr[idx_j,idx_i])),(int(i),int(j+leny*0.9)),font,0.75,(0,255,0),2)
                cv2.imshow("OCR",dst2)
                cv2.imwrite("segmented.jpg",dst2)
                cv2.waitKey(1000)
            i+=lenx
            idx_i+=1
            cv2.destroyWindow("sudo")
        j+=leny
        idx_j+=1
    print arr
    cv2.destroyAllWindows()
    cv2.imwrite("ocr.jpg",dst2)
    return arr



"""def ocr_nn2(dst,dst2,thick=1):
    Xtrain = np.loadtxt("final_feature.txt")
    ytrain = np.loadtxt("final_output.txt")
    #Xtrain = Xtrain/255.0
    #ytrain = ytrain
    X = [x.reshape(len(Xtrain[0]),1) for x in Xtrain]
    
    def value2vector(val):
        ans = np.zeros(10)
        ans[val] = 1
        return ans.reshape((10,1))
    Y = [value2vector(val) for val in ytrain]
    training_data = np.array([])
    for x,y in zip(X,Y):
        data = np.array([x,y])
        if len(training_data) == 0:
            training_data = data
        else:
            training_data = np.vstack((training_data,data))
        
    i = 0
    j = 0
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx
    endy = 8*leny
    arr = np.zeros((9,9))
    net = Network([100,30,10])
    net.SGD(training_data,100,10,3)
    cut_off = blank_square_cut_off(dst,thick)
    idx_j = 0
    a = []
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            if idx_j%3 == 0 and idx_i%3 == 0:
                img = img[4+thick:-4-thick,4+thick:-4-thick]
            elif idx_i%3 == 0:
                img = img[4+thick:-3-thick,4+thick:-4-thick]# Changeable  4:-4,5:-5
            elif idx_j%3 == 2:
                img = img[4+thick:-4-thick,3+thick:-4-thick]
            else: 
                img = img[4+thick:-3-thick,3+thick:-4-thick]
            r = img.shape[0]
            c = img.shape[1]
            img2 = img.copy()
            print img.sum()
            if img.sum() < cut_off:               # Changeable  14000 
                arr[idx_j,idx_i] = int(0)
                
            else:
                _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                max = 0
                max_idx = 0
                #print "Lenth of contour is" + str(len(contours))
                for k in range(len(contours)):
                    cnt = contours[k]
                    if cv2.contourArea(cnt) > max:
                        max = cv2.contourArea(cnt)
                        max_idx = k

                cnt = contours[max_idx]
                #print cnt
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(dst2,(int(i),int(j)),(int(i+lenx),int(j+leny)),(0,0,255),3)
                #print img.shape
                img = img[y:y+h,x:x+w]
                img = cv2.resize(img,(10,10))
                cv2.imshow("sud",img)
                b = img.ravel()
                b = b.reshape((100,1))
                #sb=b/255.0
                arr[idx_j,idx_i] = int(net.predict(b))
                print "Recognized number is " + str(arr[idx_j,idx_i])
                #print "Percent Score is " + str(knn.percentage_score()) + "\n"
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(dst2,str(int(arr[idx_j,idx_i])),(int(i),int(j+leny*0.9)),font,0.75,(0,255,0),2)
                cv2.imshow("OCR",dst2)
                cv2.imwrite("segmented.jpg",dst2)
                cv2.waitKey(1000)
            i+=lenx
            idx_i+=1
            cv2.destroyAllWindows()
        j+=leny
        idx_j+=1
    print arr
    return arr"""

def blank_square_cut_off(dst,thick):
    sums = []
    i = 0
    j = 0
    lenx = dst.shape[1]/9.0
    leny = dst.shape[0]/9.0
    endx = 8*lenx
    endy = 8*leny
    idx_j = 0
    while j <= endy:
        i = 0
        idx_i = 0
        while i <=endx:
            img = dst[j:j+leny,i:i+lenx]
            if idx_j%3 == 0 and idx_i%3 == 0:
                img = img[4+thick:-4-thick,4+thick:-4-thick]
            elif idx_i%3 == 0:
                img = img[4+thick:-3-thick,4+thick:-4-thick]# Changeable  4:-4,5:-5
            elif idx_j%3 == 2:
                img = img[4+thick:-4-thick,3+thick:-4-thick]
            else: 
                img = img[4+thick:-3-thick,3+thick:-4-thick]

                
            r = img.shape[0]
            c = img.shape[1]
            img2 = img.copy()

            _, contours,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max = 0
            max_idx = 0
            for k in range(len(contours)):
                cnt = contours[k]
                if cv2.contourArea(cnt) > max:
                    max = cv2.contourArea(cnt)
                    max_idx = k           


            try:
                 cnt = contours[max_idx]
                 #print cnt
                 x,y,w,h = cv2.boundingRect(cnt)
                 cv2.rectangle(dst2,(int(i),int(j)),(int(i+lenx),int(j+leny)),(0,255,0),3)
                 #print img.shape
                 img = img[y:y+h,x:x+w]
                 sums.append(img.sum())

            except:
                 sums.append(img.sum())
            i+=lenx
            idx_i+=1
        j+=leny
        idx_j+=1
    sums = np.sort(sums)
    diff = np.zeros(sums.shape)
    for i in range(len(sums)-1):
        diff[i] = sums[i+1] - sums[i]
    idx = np.argmax(diff)
    return (0.6
            *(sums[idx] + diff[idx]/2.0) + 0.4*(sums[-1]/2))

def merge_different_features():
    dir_name = "C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv\sudoku_images"
    file_idx = [3,4,5,6]
    file_name_X = "feature2.txt"
    file_name_y = "output2.txt"
    Xtrain = np.loadtxt(file_name_X)
    print Xtrain.shape
    ytrain = np.loadtxt(file_name_y)
    for idx in file_idx:
        file_name_X = "feature" + str(idx) + ".txt"
        file_name_y = "output" + str(idx) + ".txt"
        x = np.loadtxt(file_name_X)
        y = np.loadtxt(file_name_y)
        print x.shape
        Xtrain = np.vstack((Xtrain,x))
        ytrain = np.hstack((ytrain,y))
    np.savetxt("final_feature.txt",Xtrain)
    np.savetxt("final_output.txt",ytrain)

#merge_different_features()

