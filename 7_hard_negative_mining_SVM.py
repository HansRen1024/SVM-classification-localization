#!/usr/bin/env python
#encoding:utf-8

import numpy as np
import sklearn.svm as ssv
from sklearn.externals import joblib
from skimage.feature import hog
import glob
import os
import cv2

def trainSvm(datas, labels):
#    clf = ssv.SVC(kernel='rbf', C=51.888660461910455, gamma=6.340844354646774e-05) #10,50,100
    clf = ssv.SVC(kernel='rbf', C=36.076088, gamma=0.000075) #10,100,500
    print "Training a SVM Classifier."
    clf.fit(datas, labels)
    return clf

def loadData(path):
    fds = []
    labels = []
    num=0
    for feat_path in glob.glob(os.path.join(path, '*.feat')):
        num += 1
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])
        print "%d Dealing with %s" %(num,feat_path)
    return fds, labels

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window    
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

if __name__ == "__main__":
    normalize = True
    visualize = False
    block_norm = 'L2-Hys'
    cells_per_block = [2,2]
    pixels_per_cell = [20,20]
    orientations = 9
#    m = 'home'
    new_model_path = './models/svm_h_p_m.model'
    train_feat_path = './features/train'
    fds, labels = loadData(train_feat_path) 


#    clf = trainSvm(fds, labels)


    model_path = './models/svm_20pixel.model'
    clf = joblib.load(model_path)
    
#--------------------hard_negative_mining-------------------------------------
    negative_img_path = './train/negative'
    num = 0
    for childDir in os.listdir(negative_img_path):
        num += 1
        print "num: %d hard negative mining: %s" %(num,childDir)
        f = os.path.join(negative_img_path, childDir)
        data = cv2.imread(f)
        scales = [(200,200), (300,300), (400,400), (500, 500), (600,600), (800, 800)]
        for (winW,winH) in scales:
            for (x, y, window) in sliding_window(data, stepSize=100, windowSize=(winW,winH)):
                result = 0
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                if window.shape[0] != 200 or window.shape[1] != 200:
                    window = cv2.resize(window,(200,200),interpolation=cv2.INTER_CUBIC)
                gray = rgb2gray(window)/255.0
                window_fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
                window_fd.shape = 1,-1
                result = int(clf.predict(window_fd))
                if result == 1:
                    window_fd.shape = (2916,)
                    fds.append(window_fd)
                    labels.append(float(0))
    new_clf = trainSvm(fds, labels)
#-----------------------------------------------------------------------------
    joblib.dump(new_clf, new_model_path)
    print "Classifier saved to {}".format(new_model_path)
