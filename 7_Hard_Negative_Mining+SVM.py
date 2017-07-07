#!/usr/bin/env python
#encoding:utf-8

import numpy as np
import sklearn.svm as ssv
from sklearn.externals import joblib
from skimage.feature import hog
import random
import glob
import os
import cv2

def trainSvm(datas, labels):
    clf = ssv.SVC(kernel='rbf', C=17.255220940030252, gamma=1.2943653125547475e-06) #20,50,500,0.842850
    print "Training a SVM Classifier."
    clf.fit(datas, labels)
    return clf

def loadData(path):
    fds_all = []
#    fds = []
#    labels = []
    num=0
    for feat_path in glob.glob(os.path.join(path, '*.feat')):
        num += 1
        data = joblib.load(feat_path)
        fds_all.append(data)
#        fds.append(data[:-1])
#        labels.append(data[-1])
        print "%d Dealing with %s" %(num,feat_path)
    return fds_all

def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):   
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def savefeat(childDir, num_win, fd):
    fd_name = childDir.split('.')[0] + '_%d.feat' %num_win
    fd_path = os.path.join('./features/train_hnm/', fd_name)
    joblib.dump(fd, fd_path,compress=3)

if __name__ == "__main__":
    normalize = True
    visualize = False
    block_norm = 'L2-Hys'
    cells_per_block = [2,2]
    pixels_per_cell = [20,20]
    orientations = 9
    new_model_path = './models/svm_pso_hnm.model'
    train_feat_path = './features/train_'
    fds_all = loadData(train_feat_path) 

    model_path = './models/svm_pso.model'
    clf = joblib.load(model_path)
    
#--------------------hard_negative_mining-------------------------------------
    negative_img_path = './train/negative'
    num = 0
    for childDir in os.listdir(negative_img_path):
        num += 1
        num_win = 0
        print "num: %d hard negative mining: %s" %(num,childDir)
        f = os.path.join(negative_img_path, childDir)
        data = cv2.imread(f)
        scales = [(100, 100), (200,200), (300,300), (400,400), (500, 500), (600,600), (800, 800)]
        for (winW,winH) in scales:
            for (x, y, window) in sliding_window(data, stepSize=100, windowSize=(winW,winH)):
                result = 0
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                if window.shape[0] != 200 or window.shape[1] != 200:
                    window = cv2.resize(window,(200,200),interpolation=cv2.INTER_CUBIC)
                gray = rgb2gray(window)/255.0
                window_fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
                win_fd = window_fd.reshape(1, -1)
                result = int(clf.predict(win_fd))
                if result == 1:
                    num_win += 1
                    fd = np.concatenate((window_fd, (float(0),)))
                    fds_all.append(fd)
                    
                    savefeat(childDir, num_win, fd)

#                    fds.append(window_fd)
#                    labels.append(float(0))

    random.shuffle(fds_all)
    fds = np.numpy(fds_all)[:, :-1]
    labels = np.numpy(fds_all)[:, -1]
    new_clf = trainSvm(fds, labels)
#-----------------------------------------------------------------------------
    joblib.dump(new_clf, new_model_path)
    print "Classifier saved to {}".format(new_model_path)
