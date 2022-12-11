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

def trainAllClasses(trainingDict,C=1,gamma=0.001):

    xTrainData,yTrainData=extractData(trainingDict)
    clf=SVC(C=C,kernel="rbf",gamma=0.001,decision_function_shape="ovo")
    clf.fit(xTrainData,yTrainData.ravel())

    return clf

def scikitTest(allPara,testDict):

    # print("testing begin")

    xValue,yValue=extractData(testDict)

    yPred=allPara.predict(xValue)
    misClassified=[]
    confusionMatrix=np.zeros([5,5],dtype=np.float64)
    for k in range(0,yValue.shape[0]):
        if int(yPred[k])!=int(yValue[k]):
            misClassified.append(xValue[k])
        confusionMatrix[int(yPred[k]),int(yValue[k])]+=1
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
    print("\nTraining Time over classes by scikit: ",trainEnd-trainStart,"sec")

    confusionMatrix,misClassified=scikitTest(allPara,testData)

    misClassified=np.array(misClassified)

    np.random.shuffle(misClassified)
    for i in range(0,10):
        plt.imshow(misClassified[i].reshape([32,32,3]))
        plt.savefig(f"misB{i+1}.png")
        plt.close()
    
    os.chdir("..")
    
    # file="allPara.pickle"
    # fileobj=open(file,"wb")
    # pickle.dump(allPara,fileobj)
    # fileobj.close()
    
    file2="confusionMatrixB.pickle"
    fileobj2=open(file2,"wb")
    pickle.dump(confusionMatrix,fileobj2)
    fileobj2.close()

    # print(confusionMatrix)
    # print(findAcc(confusionMatrix))
    print("\nThe test set accuracy is ",findAcc(confusionMatrix))

main()