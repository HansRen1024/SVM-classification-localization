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
import cv2

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def getFeat(data):
    normalize = True
    visualize = False
    block_norm = 'L2-Hys'
    cells_per_block = [2,2]
    pixels_per_cell = [20,20]
    orientations = 9
    gray = rgb2gray(data)/255.0
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
    return fd

def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):   
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
        
if __name__ == "__main__":
    model_path = './models/svm_pso_85_hnm_50.model'
    clf = joblib.load(model_path)
    c = cv2.VideoCapture(0)
    while 1:
        ret, image = c.read()
        rects = []
#        image = cv2.resize(image,(500,500),interpolation=cv2.INTER_CUBIC)
        scales = [(200,200), (300,300)]
#        scales = [(200,200), (250, 250), (300,300)]
        for (winW,winH) in scales:
            for (x, y, window) in sliding_window(image, stepSize=100, windowSize=(winW,winH)):
                result = 0
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                if window.shape[0] != 200 or window.shape[1] != 200:
                    window = cv2.resize(window,(200,200),interpolation=cv2.INTER_CUBIC)
                    win_fd = getFeat(window)
                    win_fd.shape = 1,-1
                    result = int(clf.predict(win_fd))
                    if result == 1:
                        rects.append([x, y, x + winW, y + winH])
        rects = np.array(rects)
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
        minx = 10000
        miny = 10000
        maxx = 0
        maxy = 0
        for (xA, yA, xB, yB) in pick:
            if xA < minx:
                minx = xA
            if yA < miny:
                miny = yA
            if xB > maxx:
                maxx = xB
            if yB > maxy:
                maxy = yB
        if (abs(maxx - minx) < image.shape[1]) and (abs(maxy - miny) < image.shape[0]):
            cv2.rectangle(image, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        cv2.imshow("After NMS", image)
        cv2.waitKey(1)
