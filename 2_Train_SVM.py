#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Thu Jun 15 16:38:03 2017 
 
@author: hans 
"""  
import sklearn.svm as ssv  
from sklearn.externals import joblib  
import glob  
import os  
import time  
  
if __name__ == "__main__":  
    model_path = './models/svm.model'  
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
    t0 = time.time()  
#------------------------SVM--------------------------------------------------  
    clf = ssv.SVC(kernel='rbf') # for training initial model
#     clf = ssv.SVC(kernel='rbf', C=17.255220940030252, gamma=1.2943653125547475e-06) # for training svm_pso.model(origin model)
    print "Training a SVM Classifier."  
    clf.fit(fds, labels)  
    joblib.dump(clf, model_path)
#------------------------SVM--------------------------------------------------  
    t1 = time.time()  
    print "Classifier saved to {}".format(model_path)  
    print 'The cast of time is :%f seconds' % (t1-t0)
