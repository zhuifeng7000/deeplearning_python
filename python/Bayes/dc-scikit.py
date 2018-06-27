# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 08:20:23 2018

@author: 赵智广 文档分类用scikit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb   #lib lkie matplotlib
import sklearn.naive_bayes as sknb

def load_data_set():
    """
    创建数据集,都是假的 fake data set 
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#把一篇文档转化为向量
def document2vector(wordlist,document):
    returnVec = [0]*len(wordlist)
    for word in document:
        if word in wordlist:
            returnVec[wordlist.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#训练
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

#判断
def classifyNB(thisDocVector,p0V,p1V,pAb):
    p1 = sum(p1V * thisDocVector) + np.log(pAb)
    p0 = sum(p0V * thisDocVector) + np.log(1-pAb)
    if p1 > p0:
        return 1
    else: 
        return 0

if __name__ == '__main__':
    #数据
    document_list,class_list = load_data_set()
    print(document_list,'\n----\n',class_list)
    
    myVocabList = createVocabList(document_list)
    print('myVocabList=',myVocabList)
    
    
    trainMat=[]
    for postinDoc in document_list:
        trainMat.append(document2vector(myVocabList, postinDoc))
        
    print(trainMat)
    #训练
#    p0Vect,p1Vect,pAbusive = trainNB(trainMat,class_list)
#    print(p0Vect,p1Vect,pAbusive)
    #clf = sknb.MultinomialNB()
    clf = sknb.GaussianNB()
    clf.fit(trainMat,class_list)
    
    
    #测试
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(document2vector(myVocabList, testEntry))
    
    print(clf.predict(np.matrix(thisDoc)))
#    print(testEntry,'classified as: ',classifyNB(thisDoc,p0Vect,p1Vect,pAbusive))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(document2vector(myVocabList, testEntry))
    print(clf.predict(np.matrix(thisDoc)))

    print(clf.score(trainMat,class_list))