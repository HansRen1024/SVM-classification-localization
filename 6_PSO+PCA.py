from sklearn.externals import joblib
import os
#import sys
#sys.path.append('/home/xuegengjian/renhanchi/libsvm-3.22/python')
from svmutil import svm_train
import numpy as np
import glob
import random
import copy

n = 500
train_feat_path = './features/train'

birds = 20 # size of population
maxgen = 50
pos = [] # population of class
speed = []
bestpos = []
initpos = []
birdsbestpos = []
fds = []
dict_fds = []
labels = []
allbestpos = []
w = 1 # best belongs to [0.8,1.2]
c1 = 2
c2 = 2
r1 = random.uniform(0,1)
r2 = random.uniform(0,1)
m = 'pso'

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
#    joblib.dump(meanVal,'./features/PCA/meanVal_train_%s.mean' %m)
    newData=dataMat-meanVal
    return newData,meanVal

def pca(dataMat,n):
    print "Start to do PCA..."
    newData,meanVal=zeroMean(dataMat)
    
#    covMat=np.cov(newData,rowvar=0)
#    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
#    joblib.dump(eigVals,'./features/PCA/eigVals_train_%s.eig' %m,compress=3)
#    joblib.dump(eigVects,'./features/PCA/eigVects_train_%s.eig' %m,compress=3)
    
    eigVals = joblib.load('./features/PCA/eigVals_train_%s.eig' %m)
    eigVects = joblib.load('./features/PCA/eigVects_train_%s.eig' %m)

    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
#    joblib.dump(n_eigVect,'./features/PCA/n_eigVects_train_%s_%s.eig' %(m,n))
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
    initpos.append([])

def CalDis(list):
    fitness=0.0
    param = '-t 2 -v 3 -c %s -g %s' %(list[0],list[1])
    fitness = svm_train(labels, dict_fds, param)
    return fitness

for i in range(birds):          #initial all birds' pos,speed
    pos[i].append(random.uniform(10,20))
    pos[i].append(random.uniform(0.9e-06, 1.5e-06)) # 1/num_features
    speed[i].append(float(0))
    speed[i].append(float(0))
#    speed[i].append(random.uniform(-10,10))
#    speed[i].append(random.uniform(-0.00002,0.00002))
    bestpos[i] = copy.deepcopy(pos[i])
    initpos[i] = copy.deepcopy(pos[i])

def FindBirdsMostPos():
    best=CalDis(bestpos[0])
    index = 0
    for i in range(birds):
        print "\n>>>>>The %d'd time to find globel best pos. Total %d times.\n" %(i+1, birds)
        temp = CalDis(bestpos[i])
        if temp > best:
            best = temp
            index = i
            print '------- %d: %f' %(index, -best)
    return best, bestpos[index]

print "\n-------------------------Initial Globel Best Pos----------------------------------\n"
best_predict, birdsbestpos = FindBirdsMostPos()   #initial birdsbestpos
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
            if CalDis(pos[i]) > CalDis(bestpos[i]):
                bestpos[i] = copy.deepcopy(pos[i])
    best_predict, birdsbestpos = FindBirdsMostPos()
    return birdsbestpos

for asd in range(maxgen):
    print "\n>>>>>>>>The %d'd time to update parameters. Total %d times\n" %(asd+1, maxgen)
    UpdateSpeed()
    best_para = UpdatePos()
    
    allbestpos.append(best_para)
    f=open('result/PSO_%s-%s-%s.txt' %(birds,maxgen,n),'w')
    f.write(str(allbestpos))
    f.close()
    
print "After %d iterations\nthe best C is: %f\nthe best gamma is: %f" %(maxgen,best_para[0],best_para[1])
