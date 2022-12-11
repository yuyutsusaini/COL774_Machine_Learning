import sys
import numpy as np
import cvxopt as co
import pickle
import time
import os
from itertools import count
import glob

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

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

def train(xTrainData,yTrainData,classA,classB,C=1,gamma=0.001):
    
    # number of features
    n=32*32*3
    
    # xValue,yValue=extractData(trainingDict)
    posClassSize=0
    negClassSize=0

    for j in range(0,xTrainData.shape[0]):
        if yTrainData[j]==classA:
            negClassSize+=1
        if yTrainData[j]==classB:
            posClassSize+=1

    m=posClassSize+negClassSize

    xValue=np.empty([m,n])
    yValue=np.empty([m])

    currInd=0

    for j in range(0,xTrainData.shape[0]):

        if yTrainData[j]==classA:
            xValue[currInd]=xTrainData[j].reshape([3072])
            yValue[currInd]=-1
            currInd+=1

        if yTrainData[j]==classB:
            xValue[currInd]=xTrainData[j].reshape([3072])
            yValue[currInd]=1
            currInd+=1

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
    # w=((yValue*alphas).T@xValue).reshape(-1,1)
    # S=(alphas>1e-4).flatten()
    # b=yValue[S]-np.dot(xValue[S],w)

    # Finding intercept term - b
    yPred=(-2)*np.matmul(xValue,xValue.T)
    yPred=yPred.T
    yPred+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
    yPred=yPred.T
    yPred+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
    yPred=np.exp((-gamma)*yPred)
    yPred=np.multiply(yPred,alphas.reshape(xValue.shape[0],1))
    yPred=np.multiply(yPred,yValue.reshape(xValue.shape[0],1))
    yPred=np.sum(yPred,axis=0)

    inf=float('inf')
    negSupport,posSupport=-inf,inf

    for i in range(0,xValue.shape[0]):
        if yValue[i]>0:
            posSupport=min(posSupport,yPred[i])
        else:
            negSupport=max(negSupport,yPred[i])

    b =-(negSupport+posSupport)/2

    return alphas,xValue,yValue,b

def trainAllClasses(trainingDict,C=1):

    allPara=dict()
    xTrainData,yTrainData=extractData(trainingDict)
    for i in range(0,5):
        for j in range(i+1,5):
            # print("Beginning Training ",i," ",j," class")
            allPara[i,j]=train(xTrainData,yTrainData,i,j,C)
            # print("End of Training ",i," ",j," class")

    return allPara
    
def test(allPara,testDict,gamma=0.001):

    # print("testing begin")

    xValue,yValue=extractData(testDict)
    results=np.zeros([xValue.shape[0],5,2],dtype=np.float64)

    for i in range(0,5):
        for j in range(i+1,5):
            # print("beginning testing among classes ",i," ",j)
            alphas=allPara[i,j][0]
            b=allPara[i,j][3]

            # Calculating the prediction
            yPred=(-2)*np.matmul(allPara[i,j][1],xValue.T)
            yPred=yPred.T
            yPred+=np.sum(np.square(xValue),axis=1).reshape(xValue.shape[0],1)
            yPred=yPred.T
            yPred+=np.sum(np.square(allPara[i,j][1]),axis=1).reshape(allPara[i,j][1].shape[0],1)
            yPred=np.exp((-gamma)*yPred)
            yPred=np.multiply(yPred,alphas.reshape(allPara[i,j][1].shape[0],1))
            yPred=np.multiply(yPred,allPara[i,j][2].reshape(allPara[i,j][1].shape[0],1))
            yPred=np.sum(yPred,axis=0)

            # Adding the intercept term
            yPred+=b

            for k in range(0,yValue.shape[0]):
                if yPred[k]>0:
                    results[k,j,0]+=1
                    results[k,j,1]+=yPred[k]
                else:
                    results[k,i,0]+=1
                    results[k,i,1]-=yPred[k]

    confusionMatrix=np.zeros([5,5],dtype=np.float64)
    misClassified=[]
    for k in range(0,yValue.shape[0]):
        currClass=0
        for i in range(0,5):
            if results[k,i,0]>results[k,currClass,0]:
                currClass=i
            elif results[k,i,0]==results[k,currClass,0]:
                if results[k,i,1]>results[k,currClass,1]:
                    currClass=i
        if currClass!=yValue[k]:
            misClassified.append(xValue[k])
        confusionMatrix[currClass,int(yValue[k])]+=1

    return confusionMatrix,misClassified

def findAcc(confusionMatrix):
    acc=np.trace(confusionMatrix)/np.sum(confusionMatrix)
    return 100*acc

def main():
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    trainingData=pickle.load(open(trainingDir+"/train_data.pickle","rb"))
    testData=pickle.load(open(testDir+"/test_data.pickle","rb"))

    print("Model training started")
    trainStart=time.time()
    allPara=trainAllClasses(trainingData)
    trainEnd=time.time()
    print("Model training completed")
    print("\nTraining Time over all classes by  cvxopt: ",trainEnd-trainStart,"sec")

    confusionMatrix,misClassified=test(allPara,testData)
    misClassified=np.array(misClassified)

    np.random.shuffle(misClassified)
    for i in range(0,10):
        plt.imshow(misClassified[i].reshape([32,32,3]))
        plt.savefig(f"misA{i+1}.png")
        plt.close()

    os.chdir("..")
    file1="confusionMatrixA.pickle"
    fileobj1=open(file1,"wb")
    pickle.dump(confusionMatrix,fileobj1)
    fileobj1.close()

    # print(confusionMatrix)
    # print(findAcc(confusionMatrix))
    print("\nThe test set accuracy is ",findAcc(confusionMatrix),"%\n")

main()