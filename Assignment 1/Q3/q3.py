import sys
from os.path import join, isfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calcMean(x):
    return np.sum(x,axis=1)/x.shape[1]

def calcSigma(x):
    # calculating mean feature-wise
    meanOfx=calcMean(x)
    
    # calculating variance feature-wise
    variance=np.zeros(x.shape[0],dtype=np.float64)
    for i in range(x.shape[0]):
        variance[i]+=np.sum(np.square(x[i]-meanOfx[i]))/x.shape[1]

    # root of variance=standard deviation
    standardDeviation=np.sqrt(variance)

    return standardDeviation

def normalize(x):
    #mean of all x values
    meanOfx=calcMean(x)

    # calculating standard deviation
    standardDeviationOfx=calcSigma(x)

    xc=x.copy()
    # normalizing wrt mean
    xc=xc-meanOfx.reshape(x.shape[0],1)

    # normalizing wrt standard deviation
    xc=xc/standardDeviationOfx.reshape(x.shape[0],1)
    
    return xc

def hessian(theta,x):
    # hessian matrix with all zero entries initialized
    hes=np.zeros([x.shape[1],x.shape[1]],dtype=np.float64)
    
    for i in range(x.shape[0]):
         # calculating hessian for each training data and adding over all data points
        sca=np.exp(-np.sum(theta*x[i]))/np.square(1+np.exp(-np.sum(theta*x[i])))
        for j in range(x.shape[1]):
            hes[j]-=x[i][j]*sca*x[i]
            
    return hes

def delLTheta(theta,x,y):
    # delta of l(θ) with zero entries initialized
    dTh=np.zeros(x.shape[1],dtype=np.float64)
    
    for i in range(x.shape[0]):
        # summing up del of l(θ) for whole of trainig data set
        sca=y[i]-1/(1+np.exp(-np.sum(theta*x[i])))
        for j in range(x.shape[1]):
            dTh[j]+=sca*x[i][j]
            
    return dTh

def logisticRegression(x,y,errorBound=1e-15):
    # theta initialized with all zeros
    theta=np.array([0,0,0],dtype=np.float64)
    
    while True:
        # hessian calculated
        hes=hessian(theta,x)
        
        # delta of l(θ) calculated
        dTh=delLTheta(theta,x,y)
        
        # change--> H−1∇θ((θ)
        change=np.matmul(np.linalg.inv(hes),dTh)
        theta=theta-change
        if(np.linalg.norm(change)<errorBound):
            break
    return theta

def main():
    # reading command line argumens of path to training and test data set from CLI
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    # input of features taken from csv file
    x=np.genfromtxt(join(trainingDir,"X.csv"),delimiter=",").T
    
    # input of classification taken
    y=np.genfromtxt(join(trainingDir,"Y.csv"))
    
    # normalizing x
    norx=normalize(x)
    mn=calcMean(x)
    std=calcSigma(x)

    # adding an array ones at top of x
    norx=np.vstack((np.ones(x.shape[1],dtype=np.float64),norx)).T
    
    # theta learned using logistic regression over training data set
    theta=logisticRegression(norx,y)
    # print(theta)

    outputFile=open("result_3_TrainingData.txt",mode='w')
    outputFile.write("theta0 = " +str(theta[0])+'\n')
    outputFile.write("theta1 = " +str(theta[1])+'\n')
    outputFile.write("theta2 = " +str(theta[2])+'\n')
    outputFile.close()

    mn=calcMean(x)
    std=calcSigma(x)

    #plotting decision boundary
    decisionBoundary=np.linspace((1.25-mn[0])/std[0],(8.750-mn[0])/std[0],2)
    def featureX1(X0):
        return (-theta[0]-theta[1]*X0)/theta[2]
    
    # subplot for data points and decision boundary
    fig1,plt1=plt.subplots()

    # setting limit of x and y axis values
    plt1.set_xlim(1,9)
    plt1.set_ylim(1,9)
    plt1.set_xlabel("X0")
    plt1.set_ylabel("X1")
    plt1.plot((decisionBoundary*std[0])+mn[0],(featureX1(decisionBoundary)*std[1])+mn[1],label="Decision Boundary")
    
    # plotting data points
    for i in range(x.shape[1]):
        if y[i]==1:
            plt1.scatter(x[0][i],x[1][i],color="red")
        else:
            plt1.scatter(x[0][i],x[1][i],color="black")

    plt1.legend()

    # saving the logistic regression plot in file 
    fig1.savefig("logisticRegressionPlot.png")
    # plt.show()

    # taking input of test data
    xTest=np.genfromtxt(join(testDir,"X.csv"),delimiter=",").T
    
    # printing the prediction in result_3
    testFile=open("result_3.txt",mode='w')
    for i in range(xTest.shape[1]):
        Thtx=theta[0]+theta[1]*(xTest[0][i]-mn[0])/std[0]+theta[2]*(xTest[1][i]-mn[1])/std[1]
        if 1/(1+np.exp(-Thtx))<0.5:
            testFile.write(str(0)+'\n')
        else:
            testFile.write(str(1)+'\n')
    testFile.close()

main()
# [ 0.40125316  2.5885477  -2.72558849]