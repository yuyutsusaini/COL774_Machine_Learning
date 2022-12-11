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
    for i in range(x.shape[0]):
        xc[i]=xc[i]-meanOfx[i]
    
    # normalizing wrt standard deviation
    for i in range(x.shape[0]):
        xc[i]=xc[i]/standardDeviationOfx[i]
    
    return xc
def gdaPara(x,y):
    # row represents the mu corresponding to respective class
    # column represents the feature
    mu=np.zeros([x.shape[0],2],dtype=np.float64)
    
    # Co Variance Matrix
    CoV=np.zeros([x.shape[0],2],dtype=np.float64)
    CoV0=np.zeros([x.shape[0],2],dtype=np.float64)
    CoV1=np.zeros([x.shape[0],2],dtype=np.float64)

    
    # phi--> P{y==1}
    phi=0
    
    # summing up 1{y(i) = k}x(i) for all i's and k={0,1}
    for i in range(x.shape[1]):
        if y[i]==1:
            phi+=1
            mu[1]+=x[:,i]
        else:
            mu[0]+=x[:,i]
            
    # dividing by total no of examples in each class
    mu[1]/=phi
    mu[0]/=(y.size-phi)
    
    # summing up (x(i) − µy(i))(x(i) − µy(i)).T over all i's
    for i in range(x.shape[1]):
        CoV+=np.matmul((x[:,i]-mu[y[i]]).reshape(x.shape[0],1),(x[:,i]-mu[y[i]]).reshape(x.shape[0],1).T)
    CoV/=y.size
    
    for i in range(x.shape[1]):
        CoV+=np.matmul((x[:,i]-mu[y[i]]).reshape(x.shape[0],1),(x[:,i]-mu[y[i]]).reshape(x.shape[0],1).T)
    CoV/=y.size
    
    for i in range(x.shape[1]):
        if y[i]==0:
            CoV0+=np.matmul((x[:,i]-mu[y[i]]).reshape(x.shape[0],1),(x[:,i]-mu[y[i]]).reshape(x.shape[0],1).T)
    CoV0/=(y.size-phi)
    
    for i in range(x.shape[1]):
        if y[i]==1:
            CoV1+=np.matmul((x[:,i]-mu[y[i]]).reshape(x.shape[0],1),(x[:,i]-mu[y[i]]).reshape(x.shape[0],1).T)
    CoV1/=phi

    # dividing by total no of examples
    phi/=y.size
    
    return phi,mu,CoV,CoV0,CoV1


def main():
    # reading command line argumens of path to training and test data set from CLI
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    x=np.genfromtxt(join(trainingDir,"X.csv"),delimiter=",").T

    li=np.loadtxt(join(trainingDir,"Y.csv"), dtype=str)
    y=np.zeros(len(li),dtype=int)
    for i in range(len(li)):
        if li[i]=='Canada':
            y[i]=1

    # finding the φ,µ,Σ,Σ0,Σ1 for the given data
    phiA,muA,CoVA,CoV0A,CoV1A=gdaPara(normalize(x),y)
    # print(phiA,muA,CoVA,CoV0A,CoV1A,sep='\n')
    mn=calcMean(x)
    std=calcSigma(x)
    
    outputFile=open("result_4_TrainingData.txt",mode='w')
    outputFile.write("phi = " +str(phiA)+'\n')
    outputFile.write("µ0 = " +str(muA[0])+'\n')
    outputFile.write("µ1 = " +str(muA[1])+'\n')
    outputFile.write("sigma = " +str(CoVA)+'\n')
    outputFile.write("sigma0 = " +str(CoV0A)+'\n')
    outputFile.write("sigma1 = " +str(CoV1A)+'\n')
    outputFile.close()

    fig1,plt1=plt.subplots()
    for i in range(x.shape[1]):
        if y[i]==1:
            plt1.scatter(x[0][i],x[1][i],color="red")
        else:
            plt1.scatter(x[0][i],x[1][i],color="black")
    plt1.set_xlabel("Fresh water Growth Ring Diameter")
    plt1.set_ylabel("Marine water Growth Ring Diameter")
    plt1.set_title("Training Data Distribution")
    
    
    c0=np.log(phiA/(1-phiA))+(1/2)*(np.matmul(muA[0].reshape(x.shape[0],1).T,np.matmul(np.linalg.inv(CoVA),muA[0].reshape(x.shape[0],1)))-np.matmul(muA[1].reshape(x.shape[0],1).T,np.matmul(np.linalg.inv(CoVA),muA[1].reshape(x.shape[0],1))))
    m0=np.matmul(muA[1].reshape(x.shape[0],1).T,np.linalg.inv(CoVA))-np.matmul(muA[0].reshape(x.shape[0],1).T,np.linalg.inv(CoVA))
    
    # mean and standard deviation of x
    mn0=calcMean(x)
    std0=calcSigma(x)

    fig1.savefig("gdaDecsionBoundary.png")
    decisionBoundary=np.linspace((45-mn0[0])/std0[0],(195-mn0[0])/std0[0],2)

    def featureX1Linear(X0):
        return ((-c0-m0[0][0]*X0)/m0[0][1])[0]
    
    #plotting linear decision boundary
    plt1.plot((decisionBoundary*std0[0])+mn0[0],(featureX1Linear(decisionBoundary)*std0[1])+mn0[1],label="Decision Boundary")
    
    c1=np.log(phiA/(1-phiA))+(1/2)*(np.log(np.linalg.det(CoV0A)/np.linalg.det(CoV1A)))+(1/2)*(np.matmul(muA[0].reshape(x.shape[0],1).T,np.matmul(np.linalg.inv(CoV0A),muA[0].reshape(x.shape[0],1)))-np.matmul(muA[1].reshape(x.shape[0],1).T,np.matmul(np.linalg.inv(CoV1A),muA[1].reshape(x.shape[0],1))))
    m1=np.linalg.inv(CoV0A)-np.linalg.inv(CoV1A)
    m2=np.matmul(muA[1].reshape(x.shape[0],1).T,np.linalg.inv(CoV1A))-np.matmul(muA[0].reshape(x.shape[0],1).T,np.linalg.inv(CoV0A))

    # plotting parabolic decision boundary
    X,Y=np.meshgrid(np.linspace(-2.8,4,1000),np.linspace(-2.8,4,1000))
    Z=np.zeros([X.shape[0],X.shape[1]])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j]+=(1/2)*np.matmul(np.matmul(np.array([[X[i][j],Y[i][j]]]),m1),np.array([[X[i][j]],[Y[i][j]]]))[0][0]+np.matmul(m2,np.array([[X[i][j]],[Y[i][j]]]))[0][0]+c1
    plt1.contour((X*std0[0])+mn0[0],(Y*std0[1])+mn0[1],Z,0)

    # saving the gda data points and decision boundary in a file
    # plt.show()
    
    xTest=np.genfromtxt(join(testDir,"X.csv"),delimiter=",").T

    def check(testCase):
        pY0=-(1/2)*np.log(np.linalg.det(CoV0A))-(1/2)*np.matmul(np.matmul((testCase-muA[0]).reshape(x.shape[0],1).T,CoV0A),(testCase-muA[0]).reshape(x.shape[0],1))[0][0]+np.log(1-phiA)
        pY1=-(1/2)*np.log(np.linalg.det(CoV1A))-(1/2)*np.matmul(np.matmul((testCase-muA[1]).reshape(x.shape[0],1).T,CoV1A),(testCase-muA[1]).reshape(x.shape[0],1))[0][0]+np.log(phiA)
        if pY1>pY0:
            return "Canada"
        else:
            return "Alaska"

    # printing the prediction in result_4
    testFile=open("result_4.txt",mode='w')
    for i in range(xTest.shape[1]):
        # print(xTest[:,i])
        testFile.write(check((xTest[:,i]-mn)/std)+'\n')
    testFile.close()
    
main()