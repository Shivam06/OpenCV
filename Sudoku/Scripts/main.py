import numpy as np
import os
import cv2
from kNearestNeighbour import KNearestNeighbour
os.chdir("C:\Users\SHIVAM MAHAJAN\Desktop")
Xtrain = np.loadtxt("inputs_train.data")
Xtest = np.loadtxt("inputs_test.data")
ytrain = np.loadtxt("response_train.txt")
ytest = np.loadtxt("response_test.txt")
knn = KNearestNeighbour(10)
knn.fit(Xtrain,ytrain)
Xtest = 200*np.ones(100)
ytest = 1
ypred = knn.predict(Xtest)
print ypred
print "Accuracy is " +  str(knn.accuracy_score(ypred,ytest)*100) + "%."


