#!/usr/bin/env python  
#encoding:utf-8  
  
""" 
Created on Thu Jun 15 17:29:22 2017 
 
@author: hans 
"""  
  
import numpy as np  
import sklearn.svm as ssv  
from sklearn.externals import joblib  
import glob  
import os  
import time  
m = '20pixel' # HoG choses pixel per cell will gain different number of feature values.  
def zeroMean(dataMat): # zero normalisation
    meanVal=np.mean(dataMat,axis=0) # calculate mean value of every column.   
    joblib.dump(meanVal,'./features/PCA/%s/meanVal_train_%s.mean' %(m,m)) # save mean value   
    newData=dataMat-meanVal   
    return newData,meanVal
def pca(dataMat,n):   
    print "Start to do PCA..."   
    t1 = time.time()   
    newData,meanVal=zeroMean(dataMat)   
    covMat=np.cov(newData,rowvar=0)   
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) # calculate feature value and feature vector   
    joblib.dump(eigVals,'./features/PCA/%s/eigVals_train_%s.eig' %(m,m),compress=3)    
    joblib.dump(eigVects,'./features/PCA/%s/eigVects_train_%s.eig' %(m,m),compress=3)  
    # eigVals = joblib.load('./features/PCA/%s/eigVals_train_%s.eig' %(m,m))  
    # eigVects = joblib.load('./features/PCA/%s/eigVects_train_%s.eig' %(m,m))   
    eigValIndice=np.argsort(eigVals) # sort feature value
    n_eigValIndice=eigValIndice[-1:-(n+1):-1] # take n feature value   
    n_eigVect=eigVects[:,n_eigValIndice] # take n feature vector 
    joblib.dump(n_eigVect,'./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,n))    
    lowDDataMat=newData*n_eigVect # calculate low dimention data
    # reconMat=(lowDDataMat*n_eigVect.T)+meanVal   
    t2 = time.time()   
    print "PCA takes %f seconds" %(t2-t1)   
    return lowDDataMat  
if __name__ == "__main__":   
    n = 100 # this is to define how dimentions u want   
    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,n)   
    train_feat_path = './features/train'   
    fds = []   
    labels = []   
    num=0   
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):   
        num += 1   
        data = joblib.load(feat_path)   
        fds.append(data[:-1])   
        labels.append(data[-1])   
        print "%d Dealing with %s" %(num,feat_path)  
#------------------------PCA--------------------------------------------------   
    fds = np.array(fds,dtype = int) #TODO, force to int format may damage the value to be all 0.
    fds.shape = 2327,-1 # 2327 is the number of trainset  
    fds= pca(fds,n)  
#------------------------PCA--------------------------------------------------   
    t0 = time.time()  
#------------------------SVM--------------------------------------------------   
    clf = ssv.SVC(kernel='rbf')   
    print "Training a SVM Classifier."   
    clf.fit(fds, labels)   
    joblib.dump(clf, model_path)  
#------------------------SVM--------------------------------------------------   
    t1 = time.time()   
    print "Classifier saved to {}".format(model_path)   
    print 'The cast of time is :%f seconds' % (t1-t0) 
