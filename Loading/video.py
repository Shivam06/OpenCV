import numpy
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if ret:
        cv2.imshow("win1",frame)
        if cv2.waitKey(30)==27:
            break

cap.release()
cv2.destroyAllWindows()
