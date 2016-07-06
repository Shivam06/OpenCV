import cv2
import numpy as np
import os
os.chdir(r'C:\Users\SHIVAM MAHAJAN\Desktop')
img = cv2.imread('messi.jpg',cv2.IMREAD_GRAYSCALE)
# we can also use cv2.IMREAD_UNCHANGED or cv2.IMREAD_COLOR - which returns colored image without alpha channel.
# alpha channel represent degree of opaqueness.
# we can also use -1,0,1 for second parameter. -1 : unchanged, 0 : gray, 1: color.

cv2.imshow("win",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
