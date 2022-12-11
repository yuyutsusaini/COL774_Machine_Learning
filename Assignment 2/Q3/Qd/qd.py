import sys
import numpy as np
import cvxopt as co
import pickle
import time
import os

from sklearn.svm import SVC
import matplotlib.pyplot as plt

def extractData(dataDict):
    # number of features
    n=32*32*3
    
    xTrainData=np.empty([dataDict["data"].shape[0],n])
    yTrainData=np.empty([dataDict["data"].shape[0]])

    currInd=0

    for j in range(0,dataDict["data"].shape[0]):
        xTrainData[currInd]=dataDict["data"][j].reshape([3072])
        yTrainData[currInd]=(dataDict["labels"][j])
        currInd+=1

    # Values Extracted
    xTrainData=xTrainData/255
    return xTrainData,yTrainData

# def trainAllClasses(trainingDict,C=1,gamma=0.001):

#     xTrainData,yTrainData=extractData(trainingDict)
#     clf=SVC(C=C,kernel="rbf",gamma=0.001,decision_function_shape="ovo")
#     clf.fit(xTrainData,yTrainData.ravel())

#     return clf

def scikitValidate(xTrainDataShuffle,yTrainDataShuffle,xTestData,yTestData,testIndex,c):
    # print("testing begin")

    # xValue,yValue=extractData(testDict)
    # print(xTrainDataShuffle[0].shape)
    xTrainData=np.empty([0,xTestData.shape[1]])
    yTrainData=np.empty([0])
    for i in range(0,5):
        if i!=testIndex:
            xTrainData=np.concatenate((xTrainData,xTrainDataShuffle[i]),axis=0)
            yTrainData=np.concatenate((yTrainData,yTrainDataShuffle[i]),axis=0)

    # print(xTrainData.shape)
    # print(yTrainData.shape)
    clf=SVC(C=c,kernel="rbf",gamma=0.001,decision_function_shape="ovo")
    clf.fit(xTrainData,yTrainData.ravel())

    yPredTrain=clf.predict(xTrainDataShuffle[testIndex])
    yPredTest=clf.predict(xTestData)
    # print("done")
    # yPred=allPara.predict(xValue)
    accTrain=0
    accTest=0
    for i in range(0,yPredTrain.shape[0]):
        if yPredTrain[i]==yTrainData[i]:
            accTrain+=1
    for i in range(0,yPredTest.shape[0]):
        if yPredTest[i]==yTestData[i]:
            accTest+=1
    # confusionMatrix=np.zeros([5,5],dtype=np.float64)
    # for k in range(0,yValue.shape[0]):
    #     confusionMatrix[int(yPred[k]),int(yValue[k])]+=1
    return (accTrain/yPredTrain.shape[0])*100,(accTest/yPredTest.shape[0])*100

def main():

    cPossible=np.array([1e-5,1e-3,1.0,5.0,10.0])

    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    trainingData=pickle.load(open(trainingDir+"/train_data.pickle","rb"))
    testData=pickle.load(open(testDir+"/test_data.pickle","rb"))

    xTrainData,yTrainData=extractData(trainingData)
    xTestData,yTestData=extractData(testData)

    indArr=np.random.permutation(xTrainData.shape[0])

    xTrainDataShuffle=np.copy(xTrainData)
    yTrainDataShuffle=np.copy(yTrainData)

    for i in range(0,xTrainData.shape[0]):
        xTrainDataShuffle[i]=xTrainData[indArr[i]]
        yTrainDataShuffle[i]=yTrainData[indArr[i]]
    
    xTrainDataShuffle=xTrainDataShuffle.reshape([5,xTrainData.shape[0]//5,xTrainData.shape[1]])
    yTrainDataShuffle=yTrainDataShuffle.reshape([5,yTrainData.shape[0]//5])

    # print(xTrainDataShuffle.shape)
    # print(yTrainDataShuffle.shape)
    cTrainPlt=np.empty([5])
    cTestPlt=np.empty([5])
    for i in range(0,len(cPossible)):
        cTrainavg=0
        cTestavg=0
        for j in range(0,5):
            accTrain,accTest=scikitValidate(xTrainDataShuffle,yTrainDataShuffle,xTestData,yTestData,j,cPossible[i])
            cTrainavg+=accTrain
            cTestavg+=accTest
        cTrainavg/=5
        cTestavg/=5
        print("\nThe accuracy over training data set for C = ", cPossible[i]," is ",cTrainavg,"%")
        print("The accuracy over test data set for C = ", cPossible[i]," is ",cTestavg,"%")
        cTrainPlt[i]=cTrainavg
        cTestPlt[i]=cTestavg

    plt.xlim(-1, 12)
    plt.ylim(10, 70)
    plt.plot(cPossible,cTrainPlt)
    plt.plot(cPossible,cTestPlt)
    plt.legend(['Validation Accuracy','Test Accuracy'])
    plt.savefig("AccPlot.png")
    
main()