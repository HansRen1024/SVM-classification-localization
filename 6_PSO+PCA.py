#!/usr/bin/env python
#encoding:utf-8

from sklearn.externals import joblib
from svmutil import svm_train
import numpy as np
import os
import glob
import random
import copy

n = 100 # how many dimention do you want 
m = '20pixel' # if m made u confused, u can delete all about m. it is not necessary.
train_feat_path = './features/train'

birds = 30 # size of population
maxgen = 50
pos = []
speed = []
bestpos = []
birdsbestpos = []
fds = []
dict_fds = []
labels = []
w = 0.8 # best belongs to [0.8,1.2]
c1 = 2
c2 = 2
r1 = random.uniform(0,1)
r2 = random.uniform(0,1)

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    joblib.dump(meanVal,'./features/PCA/%s/meanVal_train_%s.mean' %(m,m))
    newData=dataMat-meanVal
    return newData,meanVal

def pca(dataMat,n):
    print "Start to do PCA..."
    newData,meanVal=zeroMean(dataMat)
    
    covMat=np.cov(newData,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    joblib.dump(eigVals,'./features/PCA/%s/eigVals_train_%s.eig' %(m,m),compress=3)
    joblib.dump(eigVects,'./features/PCA/%s/eigVects_train_%s.eig' %(m,m),compress=3)
    
#    eigVals = joblib.load('./features/PCA/%s/eigVals_train_%s.eig' %(m,m))
#    eigVects = joblib.load('./features/PCA/%s/eigVects_train_%s.eig' %(m,m))

    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    joblib.dump(n_eigVect,'./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,n))
    lowDDataMat=newData*n_eigVect
    return lowDDataMat

for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
    data = joblib.load(feat_path)
    fds.append(data[:-1])
    labels.append(data[-1])
fds = np.array(fds,dtype = float)
fds= pca(fds,n)
fds = np.array(fds,dtype = float)

for i in range(len(fds[:,0])):
    dict_data = dict(zip(range(len(data))[1:],fds[i,:]))
    dict_fds.append(dict_data)

for i in range(birds):
    pos.append([])
    speed.append([])
    bestpos.append([])

def CalDis(list):
    fitness=0.0
    param = '-t 2 -v 3 -c %s -g %s' %(list[0],list[1]) # -v decides n-fold cross validation
    fitness = svm_train(labels, dict_fds, param) 
    return -fitness

for i in range(birds):          #initial all birds' pos,speed
    pos[i].append(random.uniform(1,100))
    pos[i].append(random.uniform(0,0.0001)) # maximum 1/num_features
    speed[i].append(random.uniform(-10,10))
    speed[i].append(random.uniform(-0.00001,0.00001))
    bestpos[i] = copy.deepcopy(pos[i])

def FindBirdsMostPos():
    best=CalDis(bestpos[0])
    index = 0
    for i in range(birds):
        print "\n>>>>>The %d'd time to find globel best pos. Total %d times.\n" %(i+1, birds)
        temp = CalDis(bestpos[i])
        if temp < best:
            best = temp
            index = i
            print index
    return bestpos[index]

print "\n-------------------------Initial Globel Best Pos----------------------------------\n"
birdsbestpos = FindBirdsMostPos()   #initial birdsbestpos
print "\n-------------------------Done Globel Best Pos----------------------------------\n"

def NumMulVec(num,list):         #result is in list
    for i in range(len(list)):
        list[i] *= num
    return list

def VecSubVec(list1,list2):   #result is in list1
    for i in range(len(list1)):
        list1[i] -= list2[i]
    return list1

def VecAddVec(list1,list2):      #result is in list1
    for i in range(len(list1)):
        list1[i] += list2[i]
    return list1

def UpdateSpeed():
    #global speed
    for i in range(birds):
        temp1 = NumMulVec(w,speed[i][:])
        temp2 = VecSubVec(bestpos[i][:],pos[i])
        temp2 = NumMulVec(c1*r1,temp2[:])
        temp1 = VecAddVec(temp1[:],temp2)
        temp2 = VecSubVec(birdsbestpos[:],pos[i])
        temp2 = NumMulVec(c2*r2,temp2[:])
        speed[i] = VecAddVec(temp1,temp2)

def UpdatePos():
    print "Update Pos."
    global bestpos,birdsbestpos
    for i in range(birds):
        if pos[i][0]+speed[i][0] > 0 and pos[i][1]+speed[i][1] > 0: 
            VecAddVec(pos[i],speed[i])
            if CalDis(pos[i])<CalDis(bestpos[i]):
                bestpos[i] = copy.deepcopy(pos[i])
    birdsbestpos = FindBirdsMostPos()
    return birdsbestpos

for i in range(maxgen):
    print "\n>>>>>>>>The %d'd time to update parameters. Total %d times\n" %(i+1, maxgen)
    UpdateSpeed()
    best_para = UpdatePos()
print "Ater %d iterations\nthe best C is: %f\nthe best gamma is: %f" %(maxgen,best_para[0],best_para[1])
