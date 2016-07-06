import cv2
import numpy as np
import os
os.chdir(r'C:\Users\SHIVAM MAHAJAN\Desktop')

img = cv2.imread('messi.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(0,0),(150,150),(255,255,255),15)
cv2.rectangle(img,(25,25),(300,300),(0,0,0),10)
cv2.circle(img,(340,100),40,(255,89,200),-1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
cv2.polylines(img,[pts],True,(0,255,255),3) # True - for whether we want to close polygon or not
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,'openCV Tuts',(13,190),font,0.5,(0,0,0),5)

cv2.imshow("win",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
