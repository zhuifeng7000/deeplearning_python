# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:53:13 2018

@author: 赵智广
"""

import numpy as np

#A=np.array([[1,1],[7,7]])




def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def ecludSim(A,B):
    return 1.0/(1.0+np.linalg.norm(A-B))
    
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print('the {0:d} and {1:d} similarity is: {2:f}'.format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = np.linalg.svd(dataMat)
    Sig4 = np.matrix(np.eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)

        print('the {0:d} and {1:d} similarity is: {2:f}'.format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


def svdEst2(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = np.linalg.svd(dataMat)
    Sig4 = np.matrix(np.eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    #xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    xformedItems = U[:,:4].T *dataMat  #create transformed items //4*items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[:,item],\
                             xformedItems[:,j])
        print('the {0:d} and {1:d} similarity is: {2:f}'.format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

#A = loadExData()

#U,Sigma,VT = np.linalg.svd(A)

#print('U=',U)
#print('Sigma',Sigma)
#print('VT',VT)

A=np.matrix(loadExData())
A[0,1]=A[0,0]=A[1,0]=A[2,0]=4
A[3,3]=2
print(A)

##est = standEst(A,2,ecludSim,1)
#print('using ecludSim method,estimate[2,1]=',standEst(A,2,ecludSim,1))
#print('using pearsSim method,estimate[2,1]=',standEst(A,2,pearsSim,1))
#print('using cosSim method,estimate[2,1]=',standEst(A,2,cosSim,1))

#print(recommend(A,2))

B = np.matrix(loadExData2())
U,sigma,VT=np.linalg.svd(B)
print(sigma)

#coumpute how many sigmas we need
#sig2=sigma**2
#print('sig2=',sum(sig2))
#sumv=0
#for i in range(sigma.shape[0]):
#    n=i+1
#    sumv=sumv+sigma[i]**2
#    print('{0}th sumv={1}'.format(n,sumv))
#    if sumv>(sum(sig2)*0.9):
#        break
#
#for j in range(n):
#    print(sigma[j])
#        

print('============SVD Est cosSim================')
print(recommend(B,1,estMethod=svdEst))
print('============mySVD Est cosSim================')
print(recommend(B,1,estMethod=svdEst2))
print('============SVD Est ecludSim================')
print(recommend(B,1,estMethod=svdEst,simMeas=ecludSim))
print('============Stand Est================')
print(recommend(B,1,estMethod=standEst))

