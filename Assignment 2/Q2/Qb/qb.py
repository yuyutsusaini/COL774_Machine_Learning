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
    
    xValue=np.empty([posClassSize+negClassSize,n])
    yValue=np.empty([posClassSize+negClassSize])

    currInd=0

    for j in range(0,dataDict["data"].shape[0]):
        if dataDict["labels"][j]==1:
            # swapped=np.swapaxes(dataDict["data"][j],0,2)
            flattened=dataDict["data"][j].reshape([3072])
            # flattened=swapped.flatten()
            xValue[currInd]=flattened
            yValue[currInd]=-1
            currInd+=1
        if dataDict["labels"][j]==2:
            # swapped=np.swapaxes(dataDict["data"][j],0,2)
            flattened=dataDict["data"][j].reshape([3072])
            # flattened=swapped.flatten()
            xValue[currInd]=flattened
            yValue[currInd]=1
            currInd+=1

    xValue=xValue/255
    return xValue,yValue

def train(trainingDict,C=1,gamma=0.001):
    xValue,yValue=extractData(trainingDict)
    # xValue=xValue[0:10,0:5]
    m,n=xValue.shape
    yValue=yValue.reshape(-1,1)

    # Gaussian Kernel
    # Calculating the H matrix 
    H=(-2)*np.matmul(xValue,xValue.T)
    H+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
    H=H.T
    H+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
    H=H.T
    H=np.exp((-gamma)*H)
    H=np.multiply(np.matmul(yValue,yValue.T),H)

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
    # sol=cvxopt_solvers.qp(P,q,G,h,A,b)

    return alphas,xValue,yValue

def test(alphas,b,xTrainData,yTrainData,testDict,gamma=0.001):

    # Initializing results as predicition being right or wrong
    results=np.zeros([2],dtype=np.float64)
    # alphas=np.array(allPara['x'])
    
    # Extracting Data
    xValue,yValue=extractData(testDict)

    # Calculating the prediction
    yPred=(-2)*np.matmul(xTrainData,xValue.T)
    yPred=yPred.T
    yPred+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
    yPred=yPred.T
    yPred+=np.sum(np.square(xTrainData),axis=1).reshape(xTrainData.shape[0],1)
    yPred=np.exp((-gamma)*yPred)
    yPred=np.multiply(yPred,alphas.reshape(xTrainData.shape[0],1))
    yPred=np.multiply(yPred,yTrainData.reshape(xTrainData.shape[0],1))
    yPred=np.sum(yPred,axis=0)

    # Adding the intercept term
    yPred+=b

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

    print("Model training started")
    trainStart=time.time()
    alphas,xTrainData,yTrainData=train(trainingData)
    trainEnd=time.time()
    print("Model training completed")
    print("\nTraining Time: ",trainEnd-trainStart,"sec")
    # retrieving alphas --> Langrangian Parameters
    # alphas=np.array(allPara['x'])

    indexSorted=np.argsort(alphas.reshape(-1,))[-5:]
    for ind in indexSorted:
        # print("ind: ",ind)
        img=xTrainData[ind].reshape([32,32,3])
        plt.imshow(img)
        plt.savefig(f"sv{ind}.png")
        plt.close()

    inf=float('inf')
    negSupport,posSupport=-inf,inf
    gamma=0.001

    # Finding intercept term - b
    yPred=(-2)*np.matmul(xTrainData,xTrainData.T)
    yPred=yPred.T
    yPred+=np.sum(np.square(xTrainData),axis=1).reshape(xTrainData.shape[0],1)
    yPred=yPred.T
    yPred+=np.sum(np.square(xTrainData),axis=1).reshape(xTrainData.shape[0],1)
    yPred=np.exp((-gamma)*yPred)
    yPred=np.multiply(yPred,alphas.reshape(xTrainData.shape[0],1))
    yPred=np.multiply(yPred,yTrainData.reshape(xTrainData.shape[0],1))
    yPred=np.sum(yPred,axis=0)

    for i in range(0,xTrainData.shape[0]):
        if yTrainData[i]>0:
            posSupport=min(posSupport,yPred[i])
        else:
            negSupport=max(negSupport,yPred[i])

    b =-(negSupport+posSupport)/2

    # retrieving support vectors
    countSupportVectors=np.zeros([2],dtype=np.float64)
    for i in range(0,alphas.shape[0]):
        if alphas[i][0]>1e-5:
            if yTrainData[i]<0:
                countSupportVectors[0]+=1
            else:
                countSupportVectors[1]+=1

     # printing on console
    print("\nNo. of support vectors in Class 1 are ",countSupportVectors[0])
    print("No. of support vectors in Class 2 are ",countSupportVectors[1])
    print("Support Vector percentage in training examples is ",np.sum((countSupportVectors)/xTrainData.shape[0])*100,"%")

    testTestStart=time.time()
    results=test(alphas,b,xTrainData,yTrainData,testData)
    testTestEnd=time.time()
    print("\nTesting Time over testing data set: ",testTestEnd-testTestStart,"sec")
    # print(results)
    
    print("\nThe accuracy over test data in case of gaussian kernel is: ",results[0],"%\n")

main()