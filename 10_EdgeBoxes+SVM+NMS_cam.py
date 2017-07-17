#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:10:57 2017

@author: hans
"""

from imutils.object_detection import non_max_suppression
from edge_boxes_python import edge_boxes_python
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import os
import time



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
        
if __name__ == "__main__":
    model_path = './models/svm_pso_less_hnm_50.model'
    clf = joblib.load(model_path)
    c = cv2.VideoCapture(0)
    num = 0
    while 1:
        t0 = time.time()
        num += 1
        ret, image = c.read()
        rects = []
        eb = edge_boxes_python(os.path.expanduser('~') + '/HoG_SVM/cup/sf.dat')
        bbs = eb.get_edge_boxes(image)
        for (xmin, ymin, width, height, hb) in bbs[0:10]:
            xmin = int(xmin)
            ymin = int(ymin)
            width = int(width)
            height = int(height)
            win = image[ymin:ymin + height, xmin:xmin + width]
            window = cv2.resize(win,(200,200),interpolation=cv2.INTER_CUBIC)
            win_fd = getFeat(window)
            win_fd.shape = 1,-1
            result = int(clf.predict(win_fd))
            if result == 1:
                rects.append([xmin, ymin, xmin + width, ymin + height])
        if len(rects) != 0:
            rects = np.array(rects)
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        t1 = time.time()
        cv2.putText(image,'%.2f' %(1/(t1-t0)),(0,30),font,0.9,(255,255,255),2)
        cv2.imshow("After NMS", image)
        cv2.waitKey(1)
