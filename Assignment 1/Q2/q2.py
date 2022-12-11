import sys
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def stochasticGradientDescent(x,y,b,alpha=1e-3,errorBound=1e-2):
    # b is batch size
    # alpha-->learning rate
    # errorBound--> error bound in cost function
    # total no of training examples
    m=y.size
    
    totIterations=0 # no of iterations
    ind=0 # starting index of current mini batch
    theta=np.array([0,0,0],dtype=np.float64) # starting point of theta
    allTheta=[] # list for all intermediate values of theta and cost function
    
    # no of iterations on which the average loss function is plotted against total Iterations
    maxIterations=min(1000,10*m//b) 
    
    # preAvgLoss--> average loss previous to previous maxIterations
    preAvgLoss=0
    
    # currentAvgLoss--> average loss over last maxIterations
    currentAvgLoss=0
    
    
    # gradient descent
    while True:
        # checking out of bounds
        if ind+b-1>=m:
            # if batch is not of size b then we start from starting
            ind=0

        # errorTerm=(y-θT*x)
        errorTerm=y[ind:ind+b]-(theta[0]*x[0,ind:ind+b]+theta[1]*x[1,ind:ind+b]+theta[2]*x[2,ind:ind+b])

        #current value of cost function
        currentCost=np.sum(np.square(errorTerm))/(2*b)
        allTheta.append(np.array([theta[0],theta[1],theta[2]]))

        #Updating the value of currentAvgLoss
        currentAvgLoss+=currentCost
        
        errorTerm=np.array([errorTerm,errorTerm,errorTerm])
        change=np.sum(errorTerm * x[:,ind:ind+b], axis=1)
        
        # updating theta term
        newTheta=theta+(alpha/b)*change
        totIterations+=1
        
        # updating the theta to newTheta
        theta=newTheta
        
        # checking error is less than epsilon
        if totIterations%maxIterations==0:
            
            # averaging over maxIterations
            currentAvgLoss/=maxIterations
            
            # if currentAvgLoss and preAvgLoss difference is less than errorBound than break
            if abs(currentAvgLoss-preAvgLoss)<errorBound:
                break
                
#             # may decrease learning rate with increase in totIterations
#             alpha=alpha/(1+totIterations/1000)
            
            # updating the values of preAvgLoss and currentAvgLoss
            preAvgLoss=currentAvgLoss
            currentAvgLoss=0
        
        if totIterations>200000:
            break

        # updating the value of ind for next batch
        ind+=b

    return np.array(allTheta),alpha,errorBound,theta,totIterations
def main():
    # reading arguments from terminal
    testDir=sys.argv[1]

    # no of data points=n
    n=1000000
    
    # xi's are generated
    x0=np.ones(n,dtype=np.float64)
    x1=np.random.normal(3.0,2.0,n)
    x2=np.random.normal(-1.0,2.0,n)
    
    # array of xi's
    x=np.array([x0,x1,x2])
    
    # value of theta given
    thetaGiv=np.array([3,1,2],dtype=np.float64)
    
    # random noise with mean 0 and variance 2
    noise=np.random.normal(0,np.sqrt(2),n)
    
    # prediction values y = θ0 + θ1x1 + θ2x2 + ɛ
    y=thetaGiv[0]*x[0]+thetaGiv[1]*x[1]+thetaGiv[2]*x[2]+noise
    
    r1a,r1b,r1c,r1d,r1e=stochasticGradientDescent(x,y,1)
    r100a,r100b,r100c,r100d,r100e=stochasticGradientDescent(x,y,100)
    r10000a,r10000b,r10000c,r10000d,r10000e=stochasticGradientDescent(x,y,10000)
    r1000000a,r1000000b,r1000000c,r1000000d,r1000000e=stochasticGradientDescent(x,y,1000000)
    # print(r1d,r1e)
    outputFile=open("result_2_TrainingData.txt",mode='w')
    outputFile.write("Batch Size = " +str(1)+'\n')
    outputFile.write("Learning Rate = " +str(r1b)+'\n')
    outputFile.write("Error Bound = " +str(r1c)+'\n')
    outputFile.write("theta = " +str(r1d)+'\n')
    outputFile.write("No. of Iterations = " +str(r1e)+'\n')

    outputFile.write('\n'+"Batch Size = " +str(100)+'\n')
    outputFile.write("Learning Rate = " +str(r100b)+'\n')
    outputFile.write("Error Bound = " +str(r100c)+'\n')
    outputFile.write("theta = " +str(r100d)+'\n')
    outputFile.write("No. of Iterations = " +str(r100e)+'\n')

    outputFile.write('\n'"Batch Size = " +str(10000)+'\n')
    outputFile.write("Learning Rate = " +str(r10000b)+'\n')
    outputFile.write("Error Bound = " +str(r10000c)+'\n')
    outputFile.write("theta = " +str(r10000d)+'\n')
    outputFile.write("No. of Iterations = " +str(r10000e)+'\n')
    
    outputFile.write('\n'"Batch Size = " +str(1000000)+'\n')
    outputFile.write("Learning Rate = " +str(r1000000b)+'\n')
    outputFile.write("Error Bound = " +str(r1000000c)+'\n')
    outputFile.write("theta = " +str(r1000000d)+'\n')
    outputFile.write("No. of Iterations = " +str(r1000000e)+'\n')
    outputFile.close()

    theta=np.array([2.97127662,0.97532424,1.99255271],dtype=np.float64)

    q2test=np.genfromtxt("q2test.csv",delimiter=",")
    q2test=np.delete(q2test,0,axis=0)

    error=theta[0]+theta[1]*q2test[:,0]+theta[2]*q2test[:,1]-q2test[:,2]

    error=np.sum(np.square(error))
    error/=(2*q2test.shape[0])
    # print(error)


    xTest=np.genfromtxt(join(testDir,"X.csv"),delimiter=",").T
    
    testFile=open("result_2.txt",mode='w')

    for i in range(xTest.shape[1]):
        yVal=theta[0]+theta[1]*xTest[0][i]+theta[2]*xTest[1][i]
        testFile.write(str(yVal)+'\n')

    testFile.close()
# -------------------//--------------------------//----------------------------//------------
# -------------------//--------------------------//----------------------------//------------
# -------------------//--------------------------//----------------------------//------------
# -------------------//--------------------------//----------------------------//------------
    # plots for different b values
    fig1=plt.figure()
    ax1=plt.axes(projection='3d')

    ax1.plot(r1a[:,0],r1a[:,1],r1a[:,2])
    ax1.set_xlabel("Theta0")
    ax1.set_ylabel("Theta1")
    ax1.set_zlabel("Theta2")
    
    fig1.savefig("contourPlot1")


    fig2=plt.figure()
    ax2=plt.axes(projection='3d')

    ax2.plot(r100a[:,0],r100a[:,1],r100a[:,2])
    ax2.set_xlabel("Theta0")
    ax2.set_ylabel("Theta1")
    ax2.set_zlabel("Theta2")

    fig2.savefig("contourPlot2")


    fig3=plt.figure()
    ax3=plt.axes(projection='3d')

    ax3.plot(r10000a[:,0],r10000a[:,1],r10000a[:,2])
    ax3.set_xlabel("Theta0")
    ax3.set_ylabel("Theta1")
    ax3.set_zlabel("Theta2")

    fig3.savefig("contourPlot3")

    fig4=plt.figure()
    ax4=plt.axes(projection='3d')

    ax4.plot(r1000000a[:,0],r1000000a[:,1],r1000000a[:,2])
    ax4.set_xlabel("Theta0")
    ax4.set_ylabel("Theta1")
    ax4.set_zlabel("Theta2")

    fig4.savefig("contourPlot4")

    
main()
# [3.00139754 0.99161955 1.98323778]