import sys
import numpy as np
import cvxopt as co
import pickle
import time
import os

from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def extractData(dataDict):
    # number of features
    n=32*32*3
    # treating label 1 as -1 and label 2 as +1
    posClassSize=0
    negClassSize=0

    for j in range(0,dataDict["data"].shape[0]):
        if dataDict["labels"][j]==1:
            negClassSize+=1
        if dataDict["labels"][j]==2:
            posClassSize+=1
    
    xValue=np.empty([posClassSize+negClassSize,n])
    yValue=np.empty([posClassSize+negClassSize])

    currInd=0

    for j in range(0,dataDict["data"].shape[0]):
        if dataDict["labels"][j]==1:

            flattened=dataDict["data"][j].reshape([3072])
            xValue[currInd]=flattened
            yValue[currInd]=-1
            currInd+=1
        if dataDict["labels"][j]==2:

            flattened=dataDict["data"][j].reshape([3072])
            xValue[currInd]=flattened
            yValue[currInd]=1
            currInd+=1

    xValue=xValue/255
    return xValue,yValue


def sciketTrain(testDict,ker="linear",C=1,gam=0.001):
    xValue,yValue=extractData(testDict)

    clf=SVC(C=C,kernel=ker,gamma=gam)
    clf.fit(xValue,yValue.ravel()) 
    return clf


def scikitTest(allPara,dataDict):

    results=np.zeros([2],dtype=np.float64)

    xValue,yValue=extractData(dataDict)
    yPred=allPara.predict(xValue)
    for i in range(0,yValue.shape[0]):
        
        if yPred[i]*yValue[i]>0:
            results[0]+=1
        else:
            results[1]+=1
    
    results=results/np.sum(results)
    results=results*100

    return results

def main():
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    trainingData=pickle.load(open(trainingDir+"/train_data.pickle","rb"))
    testData=pickle.load(open(testDir+"/test_data.pickle","rb"))

    trainLinearStart=time.time()
    allParaLinear=sciketTrain(trainingData)
    trainLinearEnd=time.time()
    print("\nTime taken to train with Linear kernel by scikit: ",trainLinearEnd-trainLinearStart,"sec")

    trainGaussianStart=time.time()
    allParaGaussian=sciketTrain(trainingData,"rbf")
    trainGaussianEnd=time.time()
    print("Time taken to train with Gaussian kernel by scikit: ",trainGaussianEnd-trainGaussianStart,"sec")

    print("\nNo. of Support Vectors in case of linear kernel are ",allParaLinear.n_support_)
    print("Support Vector percentage in training examples in case of linear kernel is  ",(np.sum(allParaLinear.n_support_)/4000)*100,"%")
    print("\nNo. of Support Vectors in case of gaussian kernel are ",allParaGaussian.n_support_)
    print("Support Vector percentage in training examples in case of gaussian kernel is  ",(np.sum(allParaGaussian.n_support_)/4000)*100,"%")
    print("\nw for linear kernel = ",allParaLinear.coef_)
    print("\nb for linear kernel = ",allParaLinear.intercept_)

    print("\nModel with linear kernel training started")
    resultsLinear=scikitTest(allParaLinear,testData)
    print("Model with linear kernel training completed")
    print("\nThe accuracy over test data in case of linear kernel is: ",resultsLinear[0]," %")

    print("\nModel with gaussian kernel training started")
    resultsGaussian=scikitTest(allParaGaussian,testData)
    print("Model with gaussian kernel training completed")
    print("\nThe accuracy over test data in case of gaussian kernel is: ",resultsGaussian[0]," %\n")


main()