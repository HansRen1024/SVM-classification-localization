#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Thu Jun 15 16:44:53 2017 
 
@author: hans 
"""  
from sklearn.externals import joblib  
import glob  
import os  
import time  
  
if __name__ == "__main__":  
    model_path = './models/svm.model'  
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
        result = clf.predict(data_test_feat)  
        if int(result) == int(data_test[-1]):  
            num += 1  
        rate = float(num)/total  
        t1 = time.time()  
    print 'The classification accuracy is %f' %rate  
    print 'The cast of time is :%f seconds' % (t1-t0) 
