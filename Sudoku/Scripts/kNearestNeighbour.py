import numpy as np
import cv2
import os

class KNearestNeighbour:
    def __init__(self,k):
        self.K = k
        self.data = np.array([])
        

    def fit(self, X,y):
        y = y.reshape(y.shape[0],1)
        self.data = np.hstack((X,y))

    def predict(self, X):
        ypred = []
        a=0
        if X.ndim == 1:
            a = 1
        else:
            a=X.shape[0]
        for i in range(a):
            hash = {}
            if X.ndim > 1:
                val = np.sum(np.abs(self.data[:,0:-1] - X[i,:]),axis=1)
            else :
                val = np.sum(np.abs(self.data[:,0:-1] - X),axis = 1)
            #print min(val)
            set = []
            for j in range(self.data.shape[0]):
                tup = (val[j], self.data[j,-1])
                set.append(tup)
            set.sort()
            for i in range(self.K):
                hash[set[i][1]] = hash.get(set[i][1],0) + 1
            #print hash
            ypred.append(max(hash))
        return ypred

    def accuracy_score(self,ypred,ytest):
        ac = float(np.sum(ypred == ytest))/float(len(ypred))
        return ac
    
