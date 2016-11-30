import cv2
import numpy as np
import os

def imagProc(path,tilt=0):
    if tilt:
        """ If image is tilted, rotate it to get into right position. """
        img = rotate_image(path)                        
        #cv2.imwrite("without_rotation.jpg",img)
        
    else:
        """ Reading the colored image """
        img = cv2.imread(path,cv2.IMREAD_COLOR)         
        #cv2.imwrite("with_rotation.jpg",img)
        
    rows = img.shape[0]
    cols = img.shape[1]
    
    """ Converting the image into grey scale """
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
    #cv2.imshow("img1",img_gray)
    
    """ Using bilateral filter to remove the noises from image.
        (Bilateral filter helps in keeping the edges sharp.) """
    img_gray_copy = img_gray.copy()
    img_gray = cv2.bilateralFilter(img_gray,5,150,150)
    
    """ Converting the image to just two shades - black and white using adaptive threshold """
    th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)  
    dst = th.copy()
    th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)

    """ Obtaining all the contours of the image. """
    _,contours,_ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    max_idx,max_area = 0, 0
    for i in range(len(contours)):
        """ Finding the area of all the contours in a loop. """
        area = cv2.contourArea(contours[i])                  
        if (area > max_area):
            max_area = area
            max_idx = i
        else:
            continue
    """ cnt is the contour with maximum area """
    cnt = contours[max_idx]
    
    """ obtaining the coordinates of smallest rectangle bounding the contour. """
    x,y,w,h = cv2.boundingRect(cnt)
    
      
    epsilon = 0.01*cv2.arcLength(cnt,True)
    
    """" finding better and less complicated approximation of contour """
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    dst2=img[y:y+h,x:x+w]                                                       
    #if tilt == 0:
     #   print "Hit Enter to solve!"
    dst = dst[y:y+h,x:x+w]
    th3 = th3[y:y+h,x:x+w]
    #cv2.imwrite("threshold3.jpg",th3)
    #cv2.waitKey(0)
    #cv2.imshow("OCR",dst) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst,dst2

def rotate_image(path):
    """ Reading the image from input path """
    img = cv2.imread(path,cv2.IMREAD_COLOR)                  
    img2 = img.copy()
    rows,cols,h = img.shape
    """ Convert colored image to gray scale """
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)         
    cv2.imshow("Original Image",img)
    #drawing = np.zeros(img.shape,np.uint8)
    """ Converting the image to just two shades - black and white using adaptive threshold """
    th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)  
    _,contours,_ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """ FINDING THE LARGEST CONTOUR. """
    max_idx,max_area = 0, 0                                 
    for i in range(len(contours)): 
        area = cv2.contourArea(contours[i])                             
        if (area > max_area):
            max_area = area
            max_idx = i                                  
        else:
            continue
    cnt = contours[max_idx]
    """ Finding coordinates of largest contours """
    """ Finding the approximate and less omplicated contour coordinates """
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)                 
    cv2.drawContours(img2,[approx],-1,[255,0,255],5)
    cv2.imwrite("contour.jpg",img2)
    pts1 = []
    """ Storing all the contour coordinates in list pts1 """
    for points in approx:                                   
        pts1.append(points[0])
    print pts1
    pts1 = np.float32(pts1)
    pts1_new  = []
    sums = pts1.sum(axis = 1)
    difference = [pts[0] - pts[1] for pts in pts1]
    pts1_new.append(list(pts1[np.argmin(sums)]))
    pts1_new.append(list(pts1[np.argmin(difference)]))
    pts1_new.append(list(pts1[np.argmax(sums)]))
    pts1_new.append(list(pts1[np.argmax(difference)]))
    pts1_new = np.float32(pts1_new)
    """ pts2 is a list of final matrix to be generated after perspecive transform """
    print "rows" + str(rows)
    print "cols" + str(cols)
    if rows > 800 and cols > 800:
        pts2 = np.float32([[0,0],[0,cols/2],[rows/2,cols/2],[rows/2,0]])
    else:
        pts2 = np.float32([[0,0],[0,cols],[rows,cols],[rows,0]])
    """ Using perspective transform to find rotate the image to target cordinates (pts2). """
    M = cv2.getPerspectiveTransform(pts1_new,pts2)
    dst = cv2.warpPerspective(img2,M,(rows,cols))
    print "Hit Enter to solve"
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst                                               
