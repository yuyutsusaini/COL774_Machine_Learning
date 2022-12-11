import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import os
import time
import sys

outputFolder=""
def convertData(dataFile,conditionVectorizer=None,reviewVectorizer=None):
    dataFile['year']=dataFile['date'].dt.year
    dataFile['month']=dataFile['date'].dt.month
    dataFile['day']=dataFile['date'].dt.day
#     dataFile=dataFile.astype({'year': np.float64, 'day': np.float64,'usefulCount': np.float64})
    yVal=dataFile['rating']
    condition=dataFile['condition']
    review=dataFile['review']
    if conditionVectorizer==None:
        conditionVectorizer = CountVectorizer(stop_words='english')
        xCondition=conditionVectorizer.fit_transform(condition)
    else:
        xCondition=conditionVectorizer.transform(condition)
        
    if reviewVectorizer==None:
        reviewVectorizer = CountVectorizer(stop_words='english')
        xReview=reviewVectorizer.fit_transform(review)
    else:
        xReview=reviewVectorizer.transform(review)
    
    xVal = hstack([xCondition,xReview,dataFile.iloc[:,4:8]])

    return xVal.astype(np.float64),yVal.astype(np.float64),conditionVectorizer,reviewVectorizer

def buildDecisionTree(xTrain,yTrain,xValid,yValid,xTest,yTest):
    modelTree=DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
    modelTree.fit(xTrain,yTrain)
    return modelTree,findAcc(modelTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def gridSearch(xTrain,yTrain,xValid,yValid,xTest,yTest):
    paramGrid = {'max_depth': [None,10,20],  
              'min_samples_split': [2,4], 
              'min_samples_leaf':[1,2],
              'max_leaf_nodes':[None]}
            # 'max_leaf_nodes':[None,8,10,14,20]
    grid=GridSearchCV(DecisionTreeClassifier(criterion='entropy',class_weight='balanced'),paramGrid,refit = True,n_jobs=-1) 
    grid.fit(xTrain,yTrain)
    bestTree=grid.best_estimator_
    return bestTree,findAcc(bestTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def pruning(xTrain,yTrain,xValid,yValid,xTest,yTest):
    prunedTree=DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
    path=prunedTree.cost_complexity_pruning_path(xTrain,yTrain)
    ccpAlphas, impurities = path.ccp_alphas, path.impurities


    fig, ax = plt.subplots()
    ax.plot(ccpAlphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig(outputFolder+"/2_c_impurity_vs_alpha.png")
    
    listClf=[]
    for alpha in ccpAlphas:
        clf=DecisionTreeClassifier(criterion='entropy',class_weight='balanced',ccp_alpha=alpha)
        clf.fit(xTrain,yTrain)
        listClf.append(clf)
    # listClf = listClf[:-1]
    # ccpAlphas = ccpAlphas[:-1]

    node_counts = [clf.tree_.node_count for clf in listClf]
    depth = [clf.tree_.max_depth for clf in listClf]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccpAlphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccpAlphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig(outputFolder+"/2_c_nodeVsAlpha__depthVsAlpha.png")


    trainAcc = np.array([clf.score(xTrain,yTrain) for clf in listClf])
    validAcc = np.array([clf.score(xTrain,yTrain) for clf in listClf])
    testAcc = np.array([clf.score(xTrain,yTrain) for clf in listClf])

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training,validation and test sets")
    ax.plot(ccpAlphas, trainAcc, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccpAlphas, validAcc, marker="o", label="validation", drawstyle="steps-post")
    ax.plot(ccpAlphas, testAcc, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig(outputFolder+"/2_c_Accuracy.png")
    
    bestTree=listClf[np.argmax(validAcc)]
    # fig, ax = plt.subplots()
#     plt.figure(figsize=(20,10))
#     bestTree=listClf[np.argmax(validAcc)]
    # plot_tree(bestTree,
    #        #feature_names = var_columns, #Feature names
    #        class_names = ["1","2","3","4","5","6","7","8","9","10"], #Class names
    #        rounded = True,
    #        filled = True)

    # plt.show()
    return bestTree,findAcc(bestTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def buildRandomForest(xTrain,yTrain,xValid,yValid,xTest,yTest):
#     trainData=handleMissingValue(trainData,imputation)
    # print(trainData)
    paramGrid = {'n_estimators': [100,200,300],  
              'max_features': [1,2,3,4], 
              'min_samples_split':[2,4,6,8],
              'max_leaf_nodes':[None]}
    grid=GridSearchCV(RandomForestClassifier(criterion='entropy',class_weight='balanced',oob_score = True),paramGrid,refit = True,n_jobs=-1)
    grid.fit(xTrain,yTrain)
    bestTree=grid.best_estimator_
    
    return findAcc(bestTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def xgBoost(xTrain,yTrain,xValid,yValid,xTest,yTest):
    # trainData=handleMissingValue(trainData,imputation)
    # print(trainData)
    paramGrid = {'n_estimators': [50,100,150,200,250,300,350,400,450],  
              'subsample': [0.4,0.5,0.6,0.7,0.8], 
              'max_depth':[40,50,60,70]}
    grid=GridSearchCV(xgb.XGBClassifier(),paramGrid,refit = True,n_jobs=-1)
    grid.fit(xTrain,yTrain)
    bestTree=grid.best_estimator_
    
    return bestTree,findAcc(bestTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def lgbm(xTrain,yTrain,xValid,yValid,xTest,yTest):
    # trainData=handleMissingValue(trainData,imputation)
    # print(trainData)
    paramGrid = {'n_estimators': [500,1000,1500,2000],  
              'subsample': [0.4,0.5,0.6,0.7,0.8], 
              'max_depth':[40,50,60,70]}
    grid=GridSearchCV(lgb.LGBMClassifier(),paramGrid,refit = True,n_jobs=-1)
    grid.fit(xTrain,yTrain)
    bestTree=grid.best_estimator_
    
    return bestTree,findAcc(bestTree,xTrain,yTrain,xValid,yValid,xTest,yTest)

def findAcc(modelTree,xTrain,yTrain,xValid,yValid,xTest,yTest):
    accTraining=metrics.accuracy_score(yTrain,modelTree.predict(xTrain))
    accValid=metrics.accuracy_score(yValid,modelTree.predict(xValid))
    accTest=metrics.accuracy_score(yTest,modelTree.predict(xTest))
    
    return accTraining,accValid,accTest

def main():
    global outputFolder

    oriTrainData = pd.read_csv(sys.argv[1],parse_dates=['date'],na_filter=False)
    oriTrainData.fillna('', inplace=True)
    oriValidData = pd.read_csv(sys.argv[2],parse_dates=['date'],na_filter=False)
    oriValidData.fillna('', inplace=True)
    oriTestData = pd.read_csv(sys.argv[3],parse_dates=['date'],na_filter=False)
    oriTestData.fillna('', inplace=True)
    outputFolder=sys.argv[4]
    part=sys.argv[5]

    xTrain,yTrain,conditionVectorizer,reviewVectorizer=convertData(oriTrainData)
    xValid,yValid,conditionVectorizer,reviewVectorizer=convertData(oriValidData,conditionVectorizer,reviewVectorizer)
    xTest,yTest,conditionVectorizer,reviewVectorizer=convertData(oriTestData,conditionVectorizer,reviewVectorizer)

    # print("started")
    if part=='a' or part=="all":
        start=time.time()
        clf,Acc=buildDecisionTree(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_a.txt",mode='w')
        outputFile.write("Time to train the Decision Tree : "+str(end-start)+'\n')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set = "+str(testAcc)+'\n')
        outputFile.close()
        # plt.figure(figsize=(20,10))
        # plot_tree(clf,
        #     #feature_names = var_columns, #Feature names
        #     class_names = ["1","2","3","4","5","6","7","8","9","10"], #Class names
        #     rounded = True,
        #     filled = True)
        # plt.savefig(outputFolder+"/2_a_decision_tree.png")

    if part=='b' or part=="all":
        start=time.time()
        clf,Acc=gridSearch(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_b.txt",mode='w')
        outputFile.write("Time to train the Decision Tree using grid search : "+str(end-start)+'\n')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set after Grid search = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after Grid search = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after Grid search = "+str(testAcc)+'\n')
        outputFile.close()
        # plt.figure(figsize=(20,10))
        # plot_tree(clf,
        #     #feature_names = var_columns, #Feature names
        #     class_names = ["1","2","3","4","5","6","7","8","9","10"], #Class names
        #     rounded = True,
        #     filled = True)
        # plt.savefig(outputFolder+"/2_b_Best_decision_tree.png")

    if part=='c' or part=="all":
        start=time.time()
        clf,Acc=pruning(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_c.txt",mode='w')
        outputFile.write("Time to train and prune the Decision Tree : "+str(end-start)+'\n')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set after Pruning = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after Pruning = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after Pruning = "+str(testAcc)+'\n')
        outputFile.close()
        # plt.figure(figsize=(20,10))
        # plot_tree(clf,
        #     #feature_names = var_columns, #Feature names
        #     class_names = ["0","1"], #Class names
        #     rounded = True,
        #     filled = True)
        # plt.savefig(outputFolder+"/2_c_Best_pruned_tree.png")

    if part=='d' or part=="all":
        start=time.time()
        trainAcc,validAcc,testAcc=buildRandomForest(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_d.txt",mode='w')
        outputFile.write("Time to build the random forest : "+str(end-start)+'\n')
        outputFile.write("Accuracy over Training data set after using random forest = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after using random forest = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after using random forest = "+str(testAcc)+'\n')
        outputFile.close()
    
    if part=='e' or part=="all":
        start=time.time()
        clf,Acc=xgBoost(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_e.txt",mode='w')
        outputFile.write("Time to train the  Gradient Boosted Tree : "+str(end-start)+'\n')
        outputFile.write("Accuracy over Training data = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data = "+str(testAcc)+'\n')
        outputFile.close()

    if part=='f' or part=="all":
        start=time.time()
        clf,Acc=lgbm(xTrain,yTrain,xValid,yValid,xTest,yTest)
        end=time.time()
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/2_f.txt",mode='w')
        outputFile.write("Time to train the Gradient Boosted Machine : "+str(end-start)+'\n')
        outputFile.write("Accuracy over Training data = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data = "+str(testAcc)+'\n')
        outputFile.close()
    
    if part=='g' or part=='all':
        size=np.array([20000,40000,60000,80000,100000,120000,140000,160000])
        for s in size:
            if s<=oriTrainData.shape[0]:
                oriTrainData = oriTrainData.sample(n=s,replace=False)
            else:
                oriTrainData = oriTrainData.sample(n=s,replace=True)
            clf1,Acc1=buildDecisionTree(xTrain,yTrain,xValid,yValid,xTest,yTest)
            clf2,Acc2=gridSearch(xTrain,yTrain,xValid,yValid,xTest,yTest)
            clf3,Acc3=pruning(xTrain,yTrain,xValid,yValid,xTest,yTest)
            trainAcc4,validAcc4,testAcc4=buildRandomForest(xTrain,yTrain,xValid,yValid,xTest,yTest)
            clf5,Acc5=xgBoost(xTrain,yTrain,xValid,yValid,xTest,yTest)
            clf6,Acc6=lgbm(xTrain,yTrain,xValid,yValid,xTest,yTest)
            outputFile=open(outputFolder+"/2_g.txt",mode='w')
            outputFile.write("Training Data set size : "+ str(s)+'\n')
            outputFile.write("Part A\n")
            outputFile.write("Accuracy over all data sets = "+str(Acc1)+'\n')
            outputFile.write("Part B\n")
            outputFile.write("Accuracy over all data sets = "+str(Acc2)+'\n')
            outputFile.write("Part C\n")
            outputFile.write("Accuracy over all data sets = "+str(Acc3)+'\n')
            outputFile.write("Part D\n")
            outputFile.write("Accuracy over all data sets = "+str([trainAcc4,validAcc4,testAcc4])+'\n\n')
            outputFile.write("Part E\n")
            outputFile.write("Accuracy over all data sets = "+str(Acc5)+'\n')
            outputFile.write("Part F\n")
            outputFile.write("Accuracy over all data sets = "+str(Acc6)+'\n\n')
            outputFile.close()
#     print(buildDecisionTree(xTrain,yTrain,xValid,yValid,xTest,yTest))
#     print(gridSearch(xTrain,yTrain,xValid,yValid,xTest,yTest))
main()