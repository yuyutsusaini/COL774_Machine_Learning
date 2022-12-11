import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import os
import time
import sys

outputFolder=""
def handleMissingValue(dataArr,imputation="rem"):
    if imputation=="rem":
        return dataArr.dropna(axis = 0).to_numpy()
    elif imputation=="median":
        median = SimpleImputer(missing_values=np.nan, strategy='median')
        # print(type(median.fit_transform(dataArr)))
        return median.fit_transform(dataArr)
    elif imputation=="mode":
        mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        return mode.fit_transform(dataArr)
    elif imputation=="xgb":
        return dataArr.to_numpy()

def buildDecisionTree(trainData,validData,testData,imputation="rem"):
    trainData=handleMissingValue(trainData,imputation)
    modelTree=DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
    modelTree.fit(trainData[:,:5],trainData[:,5])

    return modelTree,findAcc(modelTree,trainData,validData,testData)

def gridSearch(trainData,validData,testData,imputation="rem"):
    trainData=handleMissingValue(trainData,imputation)
    paramGrid = {'max_depth': [5,10,20,40],  
              'min_samples_split': [2,4,6,8], 
              'min_samples_leaf':[1,2,4,8],
              'max_leaf_nodes':[None,2,4]}
            # 'max_leaf_nodes':[None,8,10,14,20]
    grid=GridSearchCV(DecisionTreeClassifier(criterion='entropy',class_weight='balanced'),paramGrid,refit = True,n_jobs=-1) 
    grid.fit(trainData[:,:5],trainData[:,5])
    bestTree=grid.best_estimator_
    return bestTree,findAcc(bestTree,trainData,validData,testData)

def pruning(trainData,validData,testData,imputation="rem"):
    trainData=handleMissingValue(trainData,imputation)
    prunedTree=DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
    path=prunedTree.cost_complexity_pruning_path(trainData[:,:5],trainData[:,5])
    ccpAlphas, impurities = path.ccp_alphas, path.impurities


    fig, ax = plt.subplots()
    ax.plot(ccpAlphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig(outputFolder+"/1_c_impurity_vs_alpha.png")
    
    listClf=[]
    for alpha in ccpAlphas:
        clf=DecisionTreeClassifier(criterion='entropy',class_weight='balanced',ccp_alpha=alpha)
        clf.fit(trainData[:,:5],trainData[:,5])
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
    plt.savefig(outputFolder+"/1_c_nodeVsAlpha__depthVsAlpha.png")

    trainAcc = np.array([clf.score(trainData[:,:5],trainData[:,5]) for clf in listClf])
    validAcc = np.array([clf.score(validData[:,:5],validData[:,5]) for clf in listClf])
    testAcc = np.array([clf.score(testData[:,:5],testData[:,5]) for clf in listClf])

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training,validation and test sets")
    ax.plot(ccpAlphas, trainAcc, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccpAlphas, validAcc, marker="o", label="validation", drawstyle="steps-post")
    ax.plot(ccpAlphas, testAcc, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig(outputFolder+"/1_c_Accuracy.png")
    
    bestTree=listClf[np.argmax(validAcc)]

    return bestTree,findAcc(bestTree,trainData,validData,testData)

def buildRandomForest(trainData,validData,testData,imputation="rem"):
    trainData=handleMissingValue(trainData,imputation)
    # print(trainData)
    paramGrid = {'n_estimators': [100,200,500,1000],  
              'max_features': [1,2,3,4], 
              'min_samples_split':[2,4,8,16],
              'max_leaf_nodes':[None]}
    grid=GridSearchCV(RandomForestClassifier(criterion='entropy',class_weight='balanced',oob_score = True),paramGrid,refit = True,n_jobs=-1)
    grid.fit(trainData[:,:5],trainData[:,5].reshape((-1,)))
    bestTree=grid.best_estimator_
    
    return bestTree,findAcc(bestTree,trainData,validData,testData)

def xgBoost(trainData,validData,testData,imputation="rem"):
    trainData=handleMissingValue(trainData,imputation)
    # print(trainData)
    paramGrid = {'n_estimators': [10,20,40,50],  
              'subsample': [0.1,0.2,0.3,0.4,0.5,0.6], 
              'max_depth':[4,5,6,7,8,9,10]}
    grid=GridSearchCV(xgb.XGBClassifier(),paramGrid,refit = True,n_jobs=-1)
    grid.fit(trainData[:,:5],trainData[:,5].reshape((-1,)))
    bestTree=grid.best_estimator_
    
    return bestTree,findAcc(bestTree,trainData,validData,testData)

def findAcc(modelTree,trainData,validData,testData):
    accTraining=metrics.accuracy_score(trainData[:,5],modelTree.predict(trainData[:,:5]))
    accValid=metrics.accuracy_score(validData[:,5],modelTree.predict(validData[:,:5]))
    accTest=metrics.accuracy_score(testData[:,5],modelTree.predict(testData[:,:5]))

    return accTraining,accValid,accTest

def main():
    global outputFolder

    oriTrainData = pd.read_csv(sys.argv[1],na_values='?')
    oriValidData = pd.read_csv(sys.argv[2],na_values='?')
    oriTestData = pd.read_csv(sys.argv[3],na_values='?')
    outputFolder=sys.argv[4]
    part=sys.argv[5]
    # outputFolder="output1"
    # part='all'

    # oriTrainData = pd.read_csv("dataset1/train.csv",na_values='?')
    # oriValidData = pd.read_csv("dataset1/val.csv",na_values='?')
    # oriTestData = pd.read_csv("dataset1/test.csv",na_values='?')

    validData = oriValidData.dropna(axis = 0).to_numpy()
    testData = oriTestData.dropna(axis = 0).to_numpy()

    # print(handleMissingValue(oriTrainData))
    if part=='a' or part=="all":
        clf,Acc=buildDecisionTree(oriTrainData,validData,testData)
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/1_a.txt",mode='w')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set = "+str(testAcc)+'\n')
        outputFile.close()
        plt.figure(figsize=(20,10))
        plot_tree(clf,
            #feature_names = var_columns, #Feature names
            class_names = ["0","1"], #Class names
            rounded = True,
            filled = True)
        plt.savefig(outputFolder+"/1_a_decision_tree.png")

    if part=='b' or part=="all":
        clf,Acc=gridSearch(oriTrainData,validData,testData)
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/1_b.txt",mode='w')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set after Grid search = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after Grid search = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after Grid search = "+str(testAcc)+'\n')
        outputFile.close()
        plt.figure(figsize=(20,10))
        plot_tree(clf,
            #feature_names = var_columns, #Feature names
            class_names = ["0","1"], #Class names
            rounded = True,
            filled = True)
        plt.savefig(outputFolder+"/1_b_Best_decision_tree.png")

    if part=='c' or part=="all":
        clf,Acc=pruning(oriTrainData,validData,testData)
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/1_c.txt",mode='w')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
        outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
        outputFile.write("Accuracy over Training data set after Pruning = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after Pruning = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after Pruning = "+str(testAcc)+'\n')
        outputFile.close()
        plt.figure(figsize=(20,10))
        plot_tree(clf,
            #feature_names = var_columns, #Feature names
            class_names = ["0","1"], #Class names
            rounded = True,
            filled = True)
        plt.savefig(outputFolder+"/1_c_Best_pruned_tree.png")
    
    if part=='d' or part=="all":
        clf,Acc=buildRandomForest(oriTrainData,validData,testData)
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/1_d.txt",mode='w')
        outputFile.write("Parameters of decision tree obtained are : "+str(clf.get_params())+'\n')
        outputFile.write("Out of Bad Accuracy is = "+str(clf.oob_score_)+'\n')
        outputFile.write("Accuracy over Training data set after using random forest = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after using random forest = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after using random forest = "+str(testAcc)+'\n')
        outputFile.close()

    if part=='e' or part=="all":
        totImp=["median","mode"]
        outputFile=open(outputFolder+"/1_e.txt",mode='w')
        for imp in totImp:
            clf,Acc=buildDecisionTree(oriTrainData,validData,testData,imp)
            trainAcc,validAcc,testAcc=Acc
            print(imp,trainAcc,validAcc,testAcc)
            outputFile.write("Imputation used over training data for decision tree is :"+str(imp)+'\n')
            outputFile.write("Parameters of Decision tree = "+str(clf.get_params())+'\n')
            outputFile.write("Depth of decision tree obtained are : "+str(clf.get_depth())+'\n')
            outputFile.write("No. of leaves in decision tree obtained are : "+str(clf.get_n_leaves())+'\n')
            outputFile.write("Accuracy over Training data set = "+ str(trainAcc)+'\n')
            outputFile.write("Accuracy over Validation data set = "+str(validAcc)+'\n')
            outputFile.write("Accuracy over Test data set = "+str(testAcc)+'\n\n')
        for imp in totImp:
            clf,Acc=buildRandomForest(oriTrainData,validData,testData,imp)
            trainAcc,validAcc,testAcc=Acc
            print(imp,trainAcc,validAcc,testAcc)
            # outputFile=open(outputFolder+"/1_e.txt",mode='w')
            outputFile.write("\nAccuracy over Training data set after using imputation in Random Forest as "+ str(imp) + " = "+  str(trainAcc)+'\n')
            outputFile.write("Accuracy over Validation data set after using imputation in Random Forest as "+ str(imp) + " = "+ str(validAcc)+'\n')
            outputFile.write("Accuracy over Test data set after using imputation in Random Forest as "+ str(imp) + " = "+ str(testAcc)+'\n')
        outputFile.close()
    
    if part=='f' or part=="all":
        clf,Acc=xgBoost(oriTrainData,validData,testData)
        trainAcc,validAcc,testAcc=Acc
        print(trainAcc,validAcc,testAcc)
        outputFile=open(outputFolder+"/1_f.txt",mode='w')
        outputFile.write("Accuracy over Training data set after using xgb = "+str(trainAcc)+'\n')
        outputFile.write("Accuracy over Validation data set after using xgb = "+str(validAcc)+'\n')
        outputFile.write("Accuracy over Test data set after using xgb = "+str(testAcc)+'\n')
        outputFile.close()
    
main()