import cv2
import numpy as np
import os
os.chdir(r'C:\Users\SHIVAM MAHAJAN\Desktop')
img = cv2.imread('messi.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("win",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
