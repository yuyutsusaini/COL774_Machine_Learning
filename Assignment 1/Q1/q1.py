import sys
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calcMean(x):
    return np.sum(x)/x.size

def calcSigma(x):
    # calculating variance
    variance=np.sum(np.square(x-calcMean(x)))/x.size

    # root of variance
    standardDeviation=np.sqrt(variance)

    return standardDeviation

def normalize(x):
    #mean of all x values
    meanOfx=calcMean(x)

    # calculating standard deviation
    standardDeviationOfx=calcSigma(x)
    
    # normalizing wrt mean
    x=x-meanOfx

    # normalizing wrt standard deviation
    x=x/standardDeviationOfx

    return x

def gradientDescent(x,y,alpha=1e-2,errorBound=1e-12):
    # alpha-->learning rate
    #errorBound--> error bound in cost function
    # no of test examples
    m=x.size

    # no of iterations initialized
    iterations=0

    # starting point of theta
    theta=np.array([0,0],dtype=np.float64)

    # added a row 1 to x for theta0
    x=np.vstack((np.ones(m,dtype=np.float64),x))
    
    # array for all intermediate values of theta and cost function
    allTheta=[]
    
    # gradient descent
    while True:
        # errorTerm=(y-θT*x)
        errorTerm=y-(theta[0]*x[0]+theta[1]*x[1])

        #current value of cost function
        currentCost=np.sum(np.square(errorTerm))
        currentCost/=(2*m)
        
        allTheta.append(np.array([theta[0],theta[1],currentCost]));

        # change=sumOf((y-θT*x)*xj)
        change=(errorTerm*x).sum(axis=1)
        
        # upd1ating theta term
        newTheta=theta+(alpha/m)*change
        iterations+=1

        # checking error is less than epsilon
        if abs(np.sum(np.square(y-(newTheta[0]*x[0]+newTheta[1]*x[1])))/(2*m)-currentCost)<errorBound:
            # difference in cost function less than error bound
            theta=newTheta
            break

        # upd1ating the theta to newTheta
        theta=newTheta
    return np.array(allTheta),alpha,errorBound,theta,iterations

# fullImp trains on data and reports graphs and learned theta with learning rate and error bound in a file
def fullImp(x,y):
    
    #normalize x
    norx=normalize(x)
    
    #gradient descent on normalized x and y
    allTheta,lr,er,th,itr=gradientDescent(norx,y)
    # print(lr,er,th,itr)

    # reg--> regression plot
    fig1,reg=plt.subplots()

    #plotting x and y's
    reg.scatter(x,y)
    
    # plotting the linear regression graph
    x1 = np.array(np.array([np.min(x)-0.5,np.max(x)+0.5]))

    #plotting linear regression line
    reg.plot(x1,th[1]*(x1-calcMean(x))/calcSigma(x)+th[0],label="prediction")
    reg.legend()

    # setting title for linear regression plot
    reg.set_title("Linear Regression")
    fig1.savefig("linearRegressionPlot.png")
    # plt.show()
    
    # plotting cost function graph
    # costp--> cost function plot
    fig2=plt.figure()
    costp=plt.axes(projection='3d')
    th0Val=np.linspace(-0.5, 2, 1000)
    th1Val=np.linspace(-0.7, 0.7, 1000)

    # X,Y--> grid of theta0,theta1 points
    X,Y=np.meshgrid(th0Val,th1Val)

    # Z--> cost function
    Z=np.square((Y*norx[0]+X-y[0]))
    for i in range(1,x.size):
        Z+=np.square((Y*norx[i]+X-y[i]))
    Z/=(2*x.size)

    # plotting the cost function
    costp.plot_surface(X,Y,Z)

    # setting the labels of x,y,z axis
    costp.set_xlabel("Theta0")
    costp.set_ylabel("Theta1")
    costp.set_zlabel("Cost function")

    # list of values of x,y,z coordinate at each iterations
    xCord1=[]
    yCord1=[]
    zCord1=[]

    # initialised plotting theta,cost function values at each iterations
    pnts1=costp.plot(xCord1,yCord1,zCord1,color='black',marker='o',linestyle='solid',linewidth=2,markersize=4)
    
    # update function which plots new values of theta at each iterations
    # def upd1(ind):
    #     xCord1.append(allTheta[ind][0])
    #     yCord1.append(allTheta[ind][1])
    #     zCord1.append(allTheta[ind][2])
    #     pnts1[0].set_xdata(xCord1)
    #     pnts1[0].set_ydata(yCord1)
    #     pnts1[0].set_3d_properties(zCord1)
    #     return pnts1

    # animate function which is using update to animate the gradient descent
    # animate1=animation.FuncAnimation(fig2,upd1,allTheta.shape[0],interval=200,blit=True)
    # upd1(0)

    #setting title for cost function plot
    costp.set_title("Cost Function Graph")
    fig2.savefig("costFunctionPlot.png")
    # plt.show()
    
    # plotting the contours of cost function
    fig3,cont=plt.subplots()
    cont.contour(X,Y,Z,100)

    # setting x and y label for contour plot
    cont.set_xlabel("Theta0")
    cont.set_ylabel("Theta1")
    
    # list of values of x,y,z coordinate at each iterations in contours
    xCord2=[]
    yCord2=[]
    
    # initialised plotting theta values at each iterations in contours
    pnts2=cont.plot(xCord2,yCord2,color="black",marker='o',linestyle="solid",linewidth=2,markersize=4)

    # update function which plots new values of theta at each iterations in contours
    # def upd2(ind):
    #     xCord2.append(allTheta[ind][0])
    #     yCord2.append(allTheta[ind][1])
    #     pnts2[0].set_xdata(xCord2)
    #     pnts2[0].set_ydata(yCord2)
    #     return pnts2
    
    # animate function which is using update to animate the gradient descent in contours
    # animate2=animation.FuncAnimation(fig3,upd2,allTheta.shape[0],interval=200,blit=True)
    # upd2(0)

    #setting title for contour plot
    cont.set_title("Contours in Cost Function")
    fig3.savefig("contourPlot")
    # plt.show()

    outputFile=open("result_1_TrainingData.txt",mode='w')
    outputFile.write("learning_rate = "+str(lr)+'\n')
    outputFile.write("stopping_criteria = "+str(er)+'\n')
    outputFile.write("theta0 = " +str(th[0])+'\n')
    outputFile.write("theta1 = " +str(th[1])+'\n')
    outputFile.write("total_iterations = "+str(itr)+'\n')

    _,lr0,er0,th0,itr0=gradientDescent(norx,y,0.1)
    outputFile.write('\n'+"learning_rate = "+str(lr0)+'\n')
    outputFile.write("stopping_criteria = "+str(er0)+'\n')
    outputFile.write("theta0 = " +str(th0[0])+'\n')
    outputFile.write("theta1 = " +str(th0[1])+'\n')
    outputFile.write("total_iterations = "+str(itr0)+'\n')

    _,lr1,er1,th1,itr1=gradientDescent(norx,y,0.025)
    outputFile.write('\n'+"learning_rate = "+str(lr1)+'\n')
    outputFile.write("stopping_criteria = "+str(er1)+'\n')
    outputFile.write("theta0 = " +str(th1[0])+'\n')
    outputFile.write("theta1 = " +str(th1[1])+'\n')
    outputFile.write("total_iterations = "+str(itr1)+'\n')
    outputFile.close()


    return allTheta,lr,er,th,itr,calcMean(x),calcSigma(x)

def runTest(theta,x,nor):

    testFile=open("result_1.txt",mode='w')
    for i in range(x.size):
        testFile.write(str(theta[0]+theta[1]*(x[i]-nor[0])/nor[1])+'\n')
    testFile.close()
def main():
    # reading command line argumens of path to training and test data set from CLI
    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    # taking input features values
    x=np.genfromtxt(join(trainingDir,"X.csv"))

    # taking input prediction values
    y=np.genfromtxt(join(trainingDir,"Y.csv"))

    # runnig the full implement function
    _,_,_,theta,_,mn,std=fullImp(x,y)
    
    # test data input taken
    xTest=np.genfromtxt(join(testDir,"X.csv"))
    
    # test data ran on the model trained
    runTest(theta,xTest,np.array([mn,std]))
    
main()
# 0.9966197905398105
# 0.0013401956023671431