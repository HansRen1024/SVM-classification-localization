#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Tue Jun 13 10:24:50 2017 
 
@author: hans 
"""  
  
from skimage.feature import hog  
from sklearn.externals import joblib  
import xml.dom.minidom as xdm  
import numpy as np  
import Image  
import cv2  
import os  
import time  
  
# define parameter  
normalize = True  
visualize = False  
block_norm = 'L2-Hys'  
cells_per_block = [2,2]  
pixels_per_cell = [20,20]  
orientations = 9  
  
# xml path  
train_xml_filePath = r'./train/Annotation'  
  
  
def getBox(childDir):  
    f_xml = os.path.join(train_xml_filePath, '%s.xml' %childDir.split('.')[0]) # organise path  
    xml = xdm.parse(f_xml) # load xml file  
    filename = xml.getElementsByTagName('filename')   
    filename = filename[0].firstChild.data.encode("utf-8") # read file name  
    xmin = xml.getElementsByTagName('xmin') # coordinate of top left pixel
    xmin = int(xmin[0].firstChild.data)  
    ymin = xml.getElementsByTagName('ymin')  
    ymin = int(ymin[0].firstChild.data)  
    xmax = xml.getElementsByTagName('xmax') # coordinate of down right pixel 
    xmax = int(xmax[0].firstChild.data)  
    ymax = xml.getElementsByTagName('ymax')  
    ymax = int(ymax[0].firstChild.data)  
    box = (xmin,ymin,xmax,ymax)   
    return box  
  
def getDataWithCrop(filePath,label):  
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f_im = os.path.join(filePath, childDir)  
        image = Image.open(f_im) # open the image  
        box = getBox(childDir)  
        region = image.crop(box) # cut off image
        data = np.asarray(region) # put the data of image into an N-dinimeter array  
        data = cv2.resize(data,(200,200),interpolation=cv2.INTER_CUBIC) # resize image  
        data = np.reshape(data, (200*200,3))   
        data.shape = 1,3,-1  
        fileName = np.array([[childDir]])  
        datalebels = zip(data, label, fileName) # organise data  
        Data.extend(datalebels) # pou the organised data into a list
        num += 1  
        print "%d processing: %s" %(num,childDir)  
    return Data,num  
  
def getData(filePath,label): # get the full image without cutting  
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f = os.path.join(filePath, childDir)  
        data = cv2.imread(f)  
        data = cv2.resize(data,(200,200),interpolation=cv2.INTER_CUBIC)  
        data = np.reshape(data, (200 * 200,3))  
        data.shape = 1,3,-1  
        fileName = np.array([[childDir]])  
        datalebels = zip(data, label, fileName)  
        Data.extend(datalebels)  
        num += 1  
        print "%d processing: %s" %(num,childDir)  
    return Data,num  
  
def getFeat(Data,mode): # get and save feature valuve
    num = 0  
    for data in Data:  
        image = np.reshape(data[0].T, (200, 200, 3)) 
        gray = rgb2gray(image)/255.0 # trans image to gray
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)  
        fd = np.concatenate((fd, data[1])) # add label in the end of the array
        filename = list(data[2])  
        fd_name = filename[0].split('.')[0]+'.feat' # set file name  
        if mode == 'train':  
            fd_path = os.path.join('./features/train/', fd_name)  
        else:  
            fd_path = os.path.join('./features/test/', fd_name)  
        joblib.dump(fd, fd_path,compress=3) # save data to local  
        num += 1  
        print "%d saving: %s." %(num,fd_name)  
  
def rgb2gray(im):  
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140  
    return gray  
  
if __name__ == '__main__':  
    t0 = time.time()  
      
    # deal with Positive test dataset and trainset with cutting  
    Ptrain_filePath = r'./train/positive'  
    Ptest_filePath = r'./test/positive'  
    PTrainData,P_train_num = getDataWithCrop(Ptrain_filePath,np.array([[1]]))  
    getFeat(PTrainData,'train')  
    PTestData,P_test_num = getData(Ptest_filePath,np.array([[1]]))  
    getFeat(PTestData,'test')  
      
    # deal with positive trainset without cutting  
    Pres_train_filePath = r'./train/positive_rest'  
    PresTrainData,Pres_train_num = getData(Pres_train_filePath,np.array([[1]]))  
    getFeat(PresTrainData,'train')  
      
    # deal with negative test dataset and train dataset without cutting  
    Ntrain_filePath = r'./train/negative'  
    Ntest_filePath = r'./test/negative'  
    NTrainData,N_train_num = getData(Ntrain_filePath,np.array([[0]]))  
    getFeat(NTrainData,'train')  
    NTestData,N_test_num = getData(Ntest_filePath,np.array([[0]]))  
    getFeat(NTestData,'test')  
      
    t1 = time.time()   
    print "------------------------------------------------"  
    print "Train Positive: %d" %(P_train_num + Pres_train_num)  
    print "Train Negative: %d" %N_train_num  
    print "Train Total: %d" %(P_train_num + Pres_train_num + N_train_num)  
    print "------------------------------------------------"  
    print "Test Positive: %d" %P_test_num  
    print "Test Negative: %d" %N_test_num  
    print "Test Total: %d" %(P_test_num+N_test_num)  
    print "------------------------------------------------"  
print 'The cast of time is:%f'%(t1-t0)
