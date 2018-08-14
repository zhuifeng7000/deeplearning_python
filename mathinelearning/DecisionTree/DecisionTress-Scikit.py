# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:44:10 2018

@author: 赵智广
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb   #lib lkie matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
import math
import sklearn.model_selection as skmodsel
import sklearn.ensemble as ske
import sklearn.metrics as skm 
import sklearn.tree as sktree
import datetime as dt


def loadDataSet():
    from sklearn import preprocessing
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')
    
    data=[]
    labels=[]
    for line in fr.readlines():
        token = line.strip().split('\t')
        data.append(token[:-1])
        labels.append(token[-1])
  
    dataA = np.array(data)
    labelsA = np.array(labels)
    print('dataA.shape[1]=',dataA.shape[1])
    
    for i in range(dataA.shape[1]):
        le1 = preprocessing.LabelEncoder()
        le1.fit(dataA[:,i])
        print(le1.transform(dataA[:,i]))
        dataA[:,i] = le1.transform(dataA[:,i])
    
    return dataA,labelsA      
    
#    # 解析数据，获得 features 数据
#    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
#    # 得到数据的对应的 Labels
#    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
#    
#    return lenses,lensesLabels



def TrainAllData():
    dataA,labelsA=loadDataSet()
    print(dataA)
    print(labelsA)
    
    clf = sktree.DecisionTreeClassifier(criterion='entropy')
    print(clf)
    
    clf.fit(dataA,labelsA)
    ''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print('feature_importances_: %s' % clf.feature_importances_)
    
    '''测试结果的打印'''
    y_pre = clf.predict(dataA)
    # print(x_train)
    print(y_pre)
    print(labelsA)
    print(np.mean(y_pre == labelsA))
#
#def show_pdf(clf):
#    '''
#    可视化输出
#    把决策树结构写入文件: http://sklearn.lzjqsdd.com/modules/tree.html
#
#    Mac报错：pydotplus.graphviz.InvocationException: GraphViz's executables not found
#    解决方案：sudo brew install graphviz
#    参考写入： http://www.jianshu.com/p/59b510bafb4d
#    '''
#    # with open("testResult/tree.dot", 'w') as f:
#    #     from sklearn.externals.six import StringIO
#    #     tree.export_graphviz(clf, out_file=f)
#
#    import pydotplus
#    from sklearn.externals.six import StringIO
#    dot_data = StringIO()
#    tree.export_graphviz(clf, out_file=dot_data)
#    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#    print(graph)

if __name__ == '__main__':
    dataA,labelsA=loadDataSet()
#    print(dataA)
#    print(labelsA)
    
    trainX,testX,trainY,testY = skmodsel.train_test_split(dataA,labelsA,test_size=0.2)
    print('拆分数据：', trainX,trainY,testX,testY)
    
    clf = sktree.DecisionTreeClassifier(criterion='entropy')
    print(clf)
    clf.fit(trainX,trainY)
    predictionY = clf.predict(testX)
    print('predictionY=',predictionY)
    print('testY=',testY)
    print(np.mean(predictionY==testY))
    
#    precision, recall, thresholds = skm.precision_recall_curve(testY, predictionY)
#    print(precision, recall, thresholds)
#    
    # 计算全量的预估结果
    answer = clf.predict_proba(dataA)
    print(answer)

    
    