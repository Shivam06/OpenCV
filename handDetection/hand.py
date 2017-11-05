import numpy as np
import cv2
import os

#os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop")
def nothing(x):
    pass

vc = cv2.VideoCapture(0)                                    
cv2.namedWindow("hand")
cv2.createTrackbar("hue_lower","hand",0,255,nothing)  # 1) Creating trackbar for lower hue value so as to find the desired colored object in frame. 
cv2.createTrackbar("hue_upper","hand",30,255,nothing) # Creating trackbar for upper hue value for same reason as above.
cv2.createTrackbar("saturation_lower","hand",41,255,nothing)  # Creating trackbar for lower saturation value for same reason as above.
cv2.createTrackbar("saturation_upper","hand",152,255,nothing)  # Creating trackbar for upper saturation value for same reason as above.
cv2.createTrackbar("value_lower","hand",69,255,nothing)    # Creating trackbar for lower value for same reason as above.
cv2.createTrackbar("value_upper","hand",220,255,nothing)    # Creating trackbar for upper value for same reason as above.
out = cv2.VideoWriter('video2.avi',-1, 20.0, (640,480))
while(1):
    
    ret,frame = vc.read()                                 # Reading one image frame from webcam.
    
    #cv2.imshow("frame",frame)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)     # Converting RGB system to HSV system.
    hl = cv2.getTrackbarPos("hue_lower","hand")  
    hu = cv2.getTrackbarPos("hue_upper","hand")           
    sl = cv2.getTrackbarPos("saturation_lower","hand")    
    su = cv2.getTrackbarPos("saturation_upper","hand")    
    vl = cv2.getTrackbarPos("value_lower","hand")         
    vu = cv2.getTrackbarPos("value_upper","hand")        
    hand_lower = np.array([hl,sl,vl])                         
    hand_upper = np.array([hu,su,vu])
    mask = cv2.inRange(frame_hsv,hand_lower,hand_upper)   
    ret,mask = cv2.threshold(mask,127,255,0)            # Region lying in HSV range of hand_lower and hand_upper has intensity : 255, rest 0
    kernel = np.ones((7,7),np.uint8)              
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)      # Performing Open operation (Increasing the white portion) to remove the noise from image 
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)     # Performing the Close operation (Decreasing the white portion)
    mask = cv2.bilateralFilter(mask,5,75,75)                 # Applying bilateral filter to further remove noises in image and keeping the boundary of hands sharp.
    #cv2.imshow("image2",mask)
    _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # finding the approximate contours of all closed objects in image
    drawing = np.zeros(frame.shape,np.uint8)               # 
    max=0                                                   #
    ci = 0                                                  #      
    for i in range(len(contours)):                          #
        cnt = contours[i]                                   #      Finding the contour with maximum size. (hand when kept considerably closer to webcam in comparison to face.
        area = cv2.contourArea(cnt)                         #
        if area>max:                                        # 
            max = area                                      #
            ci = i                                          #
    cnt = contours[ci]                                      # cnt is the largest contour
    epsilon = 0.25*cv2.arcLength(cnt,True)                  # Further trying to better approximate the contour by making edges sharper and using lesser number of points to approximate contour cnt.
    approx = cv2.approxPolyDP(cnt,epsilon,True)             
    hull = cv2.convexHull(cnt,returnPoints=True)            # Finding the convex hull of largest contour 
    cv2.drawContours(frame,[cnt],0,(255,0,0),3)             # storing the hull points and contours in "frame" image variable(matrix).
    cv2.drawContours(frame,[hull],0,(0,255,0),3)            #   (")
    hull = cv2.convexHull(cnt, returnPoints= False)     
    defects = cv2.convexityDefects(cnt,hull)                # Finding the defects between cnt contour and convex hull of hand.
    count = 0                                               # count is keeping track of number of defect points
    for i in range(defects.shape[0]):# count is keeping track of number of defect points
        s,e,f,d = defects[i,0]                 
        if d > 14000 and d<28000:                            # If normal distance between farthest point(defect) and contour is > 14000 and < 28000, it is the desired defect point.
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.circle(frame,far,5,[0,0,255],-1)             # draw a circle/ dot at the defect point. 
            count += 1                                       # count is keeping track of number of defect points
            print d
    #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,str(count+1),(100,100),font,1,(0,0,255),1)    # Outputting "count + 1"in "frame"and displaying the output.
    
    #out.write(frame)                                                 # To save the video
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) == 27:
        break
   
    
vc.release()
out.release()    
cv2.destroyAllWindows()

