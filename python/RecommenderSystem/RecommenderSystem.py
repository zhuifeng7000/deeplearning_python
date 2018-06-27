# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:36:58 2018

@author: 赵智广
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy.optimize import minimize
import sys

data = loadmat('data/ex8_movies.mat')
Y = data['Y']
R = data['R']
print(Y.shape,R.shape)
#绘图
#figure,aixes=plt.subplots(figsize=(12,12))
#aixes.imshow(Y)
#figure.tight_layout()
#plt.show()

#cost funciton
#params是线性传递过来的X和tehta
def cost(params,Y,R,num_features,lamda=0):
    Y = np.matrix(Y)
    R = np.matrix(R)
    
    num_movies = Y.shape[0] #1682
    num_users = Y.shape[1]  #943
    
    #resharp X和 theta
    X = np.matrix(np.reshape(params[:num_movies*num_features],(num_movies,num_features)))
    Theta = np.matrix(np.reshape(params[num_movies*num_features:],(num_users,num_features)))
    
    #initialize J
    J=0
    
    #compute the cost
    h=X * Theta.T
    error = np.multiply(h-Y,R)   #1682*943
    J = (1. /2) * np.sum(np.power(error,2))
    #regularization
    J = J + (lamda/2)*(np.sum(np.power(X,2))) + (lamda/2)*(np.sum(np.power(Theta,2)))
    
    
    #compute the gradient
    GradX = error * Theta + lamda * X  #1682*10
    GradTheta = error.T * X + lamda * Theta  #943*10
    
    #unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(GradX),np.ravel(GradTheta)))
    
    return J,grad
    

#params_data = loadmat('data/ex8_movieParams.mat')
#X = params_data['X']
#Theta = params_data['Theta']
#print(X.shape,Theta.shape,sep='\n')

##test
#users=4
#movies=5
#features=3
#
#X_sub = X[:movies,:features]
#Theta_sub=Theta[:users,:features]
#Y_sub = Y[:movies, :users]
#R_sub = R[:movies, :users]
#
#params = np.concatenate((np.ravel(X_sub),np.ravel(Theta_sub)))
#print(params.shape)
#
#J,grad=cost(params,Y_sub,R_sub,features,1.5)
#print('test cost=',J)
#print('test grad=',grad)  
    
movie_idx={}
f = open('data/movie_ids1.txt',encoding='gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0])-1] = ' '.join(tokens[1:])
    

for i in range(10):
    print(movie_idx[i])
    

myratings = np.zeros(Y.shape[0])
myratings[0] = 4
myratings[6] = 3
myratings[11] = 5
myratings[53] = 4
myratings[63] = 5
myratings[65] = 3
myratings[68] = 5
myratings[97] = 2
myratings[182] = 4
myratings[225] = 5
myratings[354] = 5

#insert into Y,R
print(Y.shape,R.shape)
Y = np.insert(Y,0,myratings,axis=1)
R = np.insert(R,0,myratings!=0,axis=1)
print(Y.shape,R.shape)
print(Y[0:12,0])

#begin to train
movies = Y.shape[0]
users = Y.shape[1]
features=50
lamda=10
X = np.random.standard_normal(size=(movies,features))
Theta = np.random.standard_normal(size=(users,features))
#X = np.random.random(size=(movies,features))
#Theta = np.random.random(size=(users,features))

#ravel X,Theta
parames = np.concatenate((np.ravel(X),np.ravel(Theta)))
#normalize ratings
Ynorm = np.zeros((movies,users))
Ymean = np.zeros((movies,1))


for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    
print(Ynorm.shape,Ynorm.mean())

fmin = minimize(fun=cost,x0=parames,args=(Ynorm,R,features,lamda),method='CG',
                jac=True,options={'maxiter':100})
#fmin.x是返回的值
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

#compute the prediction for me 
predictions = X*Theta.T
print(Ymean[0:20,:])
my_predictions = predictions[:,0]+Ymean
print(predictions.shape)
#sort
sorted_predictions = np.sort(my_predictions,axis=0)[::-1]#降序
print(sorted_predictions[:10])
idx = np.argsort(my_predictions, axis=0)[::-1]

print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_predictions[j])), movie_idx[j]))
