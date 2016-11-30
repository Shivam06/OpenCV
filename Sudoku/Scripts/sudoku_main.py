import cv2
import numpy as np
import os
os.chdir(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv")
import image_processing.imageProc as imageProc
from sudoku_solver.solveSudoku2 import * 
from OCR.miscellaneous_sudoku import *
os.chdir(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\Kaggle\Kaggle Data\MNIST\neural-networks-and-deep-learning-master\src")
#training_data,validation_data,test_data = mnist_loader.load_data_wrapper()


os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\opencv3.1\python opencv")
dst,dst2 = imageProc.imagProc(r"C:\Users\SHIVAM MAHAJAN\Desktop\Desktop Apps\OpenCV Images\one\sudoku25_tilt.jpg",tilt = 1)            #--
cv2.imwrite("processed_image.jpg",dst2)
#dst = dst[15:-15,:]
#cv2.imshow("d",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
arr = ocr_knn(dst,dst2.copy(),1,2)

val = solve_all([change2string(arr)])

if type(val) != bool :
    arr2 = dict2arr(val)
    #cv2.imshow("original_image",dst2)
    print "Congratulations sudoku has been recognized perfectly!"
    cv2.waitKey(0)
    cv2.waitKey(0)
    show_solved_sudoku(dst,dst2,arr2,2)
    
else:
    print "Sorry!. Sudoku can't be solved"

