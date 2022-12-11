from itertools import count
import sys
import numpy as np
import cvxopt as co
import pickle
import time
import os

from sklearn.svm import SVC
import matplotlib.pyplot as plt
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
    # print(type(dataDict["data"]))
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

def train(trainingDict,C=1):
    
    xValue,yValue=extractData(trainingDict)
    m,n=xValue.shape
    yValue=yValue.reshape(-1,1)
    xDash=yValue*xValue
    H=np.matmul(xDash,xDash.T)

    #Converting into cvxopt format
    P=cvxopt_matrix(H)
    q=cvxopt_matrix(-np.ones((m, 1)))
    G=cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h=cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A=cvxopt_matrix(yValue.reshape(1, -1))
    b=cvxopt_matrix(np.zeros(1))

    #Run solver
    sol=cvxopt_solvers.qp(P,q,G,h,A,b,options={"show_progress":False})
    # sol=cvxopt_solvers.qp(P,q,G,h,A,b)

    alphas=np.array(sol['x'])
    # w=((yValue*alphas).T@xValue).reshape(-1,1)
    # S=(alphas>1e-4).flatten()
    # b=yValue[S]-np.dot(xValue[S],w)
    return alphas,xValue,yValue

def test(alphas,b,xTrainData,yTrainData,testDict):

    # alphas=np.array(allPara['x'])
    results=np.zeros([2],dtype=np.float64)

    xValue,yValue=extractData(testDict)

    yPred=np.matmul(xTrainData,xValue.T)
    yPred=np.multiply(yPred,alphas.reshape(xTrainData.shape[0],1))
    yPred=np.multiply(yPred,yTrainData.reshape(xTrainData.shape[0],1))
    yPred=np.sum(yPred,axis=0)
    yPred+=b

    for i in range(0,yValue.shape[0]):
        if yPred[i]*yValue[i]>0:
            results[0]+=1
        else:
            results[1]+=1
    
    # print(results)
    results=results/np.sum(results)
    results=results*100

    return results

def main():
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    trainingData=pickle.load(open(trainingDir+"/train_data.pickle","rb"))
    testData=pickle.load(open(testDir+"/test_data.pickle","rb"))

    print("\nModel training started")
    trainStart=time.time()
    alphas,xTrainData,yTrainData=train(trainingData)
    # print("alphas.shape: ",alphas.shape)
    trainEnd=time.time()
    print("Model training completed")
    print("\nTraining Time: ",trainEnd-trainStart,"sec")

    indexSorted=np.argsort(alphas.reshape(-1,))[-5:]
    for ind in indexSorted:
        # print("ind: ",ind)
        img=xTrainData[ind].reshape([32,32,3])
        plt.imshow(img)
        plt.savefig(f"sv{ind}.png")
        plt.close()

    # retrieving support vectors
    # retrieving alphas --> Langrangian Parameters
    # alphas=np.array(allPara['x'])
    w=((yTrainData*alphas).T@xTrainData).reshape(-1,1)
    S=(alphas>1e-4).flatten()
    b=(yTrainData[S]-np.dot(xTrainData[S],w))[0]

    plt.imshow(w.reshape([32,32,3]))
    plt.savefig("w.png")
    plt.close()
    
    countSupportVectors=np.zeros([2],dtype=np.float64)
    for i in range(0,alphas.shape[0]):
        if alphas[i][0]>1e-5:
            if yTrainData[i]<0:
                countSupportVectors[0]+=1
            else:
                countSupportVectors[1]+=1

    # print(countSupportVectors)
    print("\nNo. of support vectors in Class 1 are ",countSupportVectors[0])
    print("No. of support vectors in Class 2 are ",countSupportVectors[1])
    print("\nSupport Vector percentage in training examples is ",np.sum((countSupportVectors)/xTrainData.shape[0])*100,"%")

    print('\nw for linear kernel = ',w)
    print('\nb for linear kernel = ',b)

    testTrainStart=time.time()
    resultsTraining=test(alphas,b,xTrainData,yTrainData,trainingData)
    testTrainEnd=time.time()
    print("\nTesting Time over training data set: ",testTrainEnd-testTrainStart,"sec")

    testTestStart=time.time()
    resultsTest=test(alphas,b,xTrainData,yTrainData,testData)
    testTestEnd=time.time()
    print("\nTesting Time over testing data set: ",testTestEnd-testTestStart,"sec")

    print("\nThe accuracy over training data is: ",resultsTraining[0],"%")
    print("The accuracy over test data is: ",resultsTest[0],"%\n")

main()

