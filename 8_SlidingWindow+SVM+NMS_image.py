#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:10:57 2017

@author: hans
"""

from imutils.object_detection import non_max_suppression
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
import time
import cv2

normalize = True
visualize = False
block_norm = 'L2-Hys'
cells_per_block = [2,2]
pixels_per_cell = [20,20]
orientations = 9

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def getFeat(data):
    gray = rgb2gray(data)/255.0
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
    return fd

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window    
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
        
if __name__ == "__main__":
#    image_path = str(raw_input("Please enter the path of an image: "))
    image_path = 'test/positive/n03147509_3599.JPEG'
    t0 = time.time()
    model_path = './models/svm_pso.model'
    clf = joblib.load(model_path)
    
    image = cv2.imread(image_path)
    image = cv2.resize(image,(500,500),interpolation=cv2.INTER_CUBIC)
    orig = image.copy()
    orig = cv2.resize(orig,(500,500),interpolation=cv2.INTER_CUBIC)
    rects = []
    scales = [(200,200), (300,300), (400, 400), (image.shape[1],image.shape[0])]
    for (winW,winH) in scales:
        for (x, y, window) in sliding_window(image, stepSize=90, windowSize=(winW,winH)):
            result = 0
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            cv2.imshow("asd", window)
            cv2.waitKey(0)
            print window.shape
            if window.shape[0] != 200 or window.shape[1] != 200:
                window = cv2.resize(window,(200,200),interpolation=cv2.INTER_CUBIC)
            win_fd = getFeat(window)
            win_fd.shape = 1,-1
            result = int(clf.predict(win_fd))
            print 'smamll image result is %d' %result
            if result == 1:
                rects.append([x, y, x + winW, y + winH])
                cv2.rectangle(orig, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
    rects = np.array(rects)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    t1 = time.time()
    print 'The cast of time is :%f seconds' % (t1-t0)
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
