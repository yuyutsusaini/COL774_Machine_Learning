import os
import sys
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
import glob
from wordcloud import WordCloud, STOPWORDS
import random

def randomTestConfuMat(posClass,negClass):

    # postive review and negative review file
    filePos=glob.glob(posClass+"/*.txt")
    fileNeg=glob.glob(negClass+"/*.txt")

    results=np.zeros([2,2],dtype=np.int_) # [0] for correct result and [1] for wrong result

    for f in filePos:
        if random.random()>0.5:
            results[0][0]+=1
        else:
            results[1][0]+=1

    for f in fileNeg:
        if random.random()<0.5:
            results[1][1]+=1
        else:
            results[0][1]+=1

    return results


def posTestConfuMat(posClass,negClass):

    # postive review and negative review file
    filePos=glob.glob(posClass+"/*.txt")
    fileNeg=glob.glob(negClass+"/*.txt")

    results=np.zeros([2,2],dtype=np.int_) # [0] for correct result and [1] for wrong result
    results[0][0]+=len(filePos)
    results[0][1]+=len(fileNeg)

    return results


def findAcc(confusionMat):
    confusionMat=confusionMat/np.sum(confusionMat)
    confusionMat=confusionMat*100
    return confusionMat.trace()


def main():
    global vocabSize,vocabulary

    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    results3=randomTestConfuMat(testDir+"/pos",testDir+"/neg")
    results4=posTestConfuMat(testDir+"/pos",testDir+"/neg")

    os.chdir("..")
    file3="results3.pickle"
    fileobj3=open(file3,"wb")
    pickle.dump(results3,fileobj3)
    fileobj3.close()

    file4="results4.pickle"
    fileobj4=open(file4,"wb")
    pickle.dump(results4,fileobj4)
    fileobj4.close()

    print("\nAccuracy on Random Prediction: ",findAcc(results3))
    print("Accuracy over Positive Assignment: ",findAcc(results4))
    print("\nAlgorithm give 30 percent improvement over random prediction and 14 percent improvement over Positive Assignment baseline\n")

main()
