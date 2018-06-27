# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:01:34 2018

@author: 赵智广

用scikit-learn的adaboots算法来判断生病的马
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb   #lib lkie matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.model_selection import cross_val_score 
import sklearn.ensemble as ske
import sklearn.metrics as skm 
import sklearn.tree as sktree

def load_data_set(file_name):
    """
    加载马的疝气病的数据
    :param file_name: 文件名
    :return: 必须要是np.array或者np.matrix不然后面没有，shape
    """
    num_feat = len(open(file_name).readline().split('\t'))
    data_arr = []
    label_arr = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr



if __name__=='__main__':
    dataM,class_Arr = load_data_set('horseColicTraining2.txt')
    print(dataM.shape)
    
    #define stump function
    stump = sktree.DecisionTreeClassifier(max_depth=1)
    
    clf = ske.AdaBoostClassifier(base_estimator=stump,n_estimators=40)
#    scores = cross_val_score(clf,dataM,class_Arr)
#    print(scores)
    clf.fit(dataM,class_Arr)
    testM,test_Arr = load_data_set('horseColicTest2.txt')
    test_rst = clf.predict(testM)
    print('error nums =',skm.zero_one_loss(test_Arr,test_rst,normalize=False))
    #accuracy=clf.score(testM,test_Arr)
    #print(accuracy)    

