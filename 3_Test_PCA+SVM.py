#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Thu Jun 15 17:29:22 2017 
 
@author: hans 
"""  
  
from sklearn.externals import joblib  
import glob  
import os  
import time  
  
m = '20pixel'  
n = 100  
  
if __name__ == "__main__":  
    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,n)  
    test_feat_path = './features/test'  
    total=0  
    num=0  
    t0 = time.time()  
    clf = joblib.load(model_path)  
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):  
        total += 1  
        print "%d processing: %s" %(total, feat_path)  
        data_test = joblib.load(feat_path)  
        data_test_feat = data_test[:-1].reshape((1,-1))  
#------------------------PCA--------------------------------------------------  
        meanVal = joblib.load('./features/PCA/%s/meanVal_train_%s.mean' %(m,m))  
        data_test_feat = data_test_feat - meanVal 
        n_eigVects = joblib.load('./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,n))  
        data_test_feat = data_test_feat * n_eigVects 
#------------------------PCA--------------------------------------------------  
        result = clf.predict(data_test_feat)  
        if int(result) == int(data_test[-1]):  
            num += 1  
        rate = float(num)/total  
        t1 = time.time()  
    print 'The classification accuracy is %f' %rate  
    print 'The cast of time is :%f seconds' % (t1-t0) 
