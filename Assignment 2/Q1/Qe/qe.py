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

stemmer=PorterStemmer()

vocabSize=0
vocabulary=dict()

def calcVocab(filePos,fileNeg,doStem=False,isBigram=False,isTriGram=False):

    global vocabSize,vocabulary
    posWords=[]
    negWords=[]

    for f in filePos:
        lines=splitToWords(f,doStem,isBigram,isTriGram) # spilling the words and removing punctuations
        for i in range(0,len(lines)):
            posWords.append(lines[i])
            if not lines[i] in vocabulary:
                vocabulary[lines[i]]=vocabSize
                vocabSize+=1
    # traversing negative review file
    for f in fileNeg:
        lines=splitToWords(f,doStem,isBigram,isTriGram) # spilling the words and removing punctuations
        for i in range(0,len(lines)):
            negWords.append(lines[i])
            if not lines[i] in vocabulary:
                vocabulary[lines[i]]=vocabSize
                vocabSize+=1

    return posWords,negWords

def train(posClass,negClass,doStem=False,isBigram=False,isTriGram=False,alpha=1):
    # Vocabulary size
    global vocabSize,vocabulary

    # postive review and negative review file
    filePos=glob.glob(posClass+"/*.txt")
    fileNeg=glob.glob(negClass+"/*.txt")

    posWords,negWords=calcVocab(filePos,fileNeg,doStem,isBigram,isTriGram)

    freqWords=np.zeros((2, vocabSize),dtype=np.float64)

    # parameters for Naive Bayes
    phi=np.array([len(filePos),len(fileNeg)],dtype=np.float64)

    # traversing positive review file
    for word in posWords:
        if word in vocabulary:
            freqWords[0][vocabulary[word]]+=1

    # traversing negative review file
    for word in negWords:
        if word in vocabulary:
            freqWords[1][vocabulary[word]]+=1

    phi=np.log(phi)-np.log(np.sum(phi))

    totWords=np.log(np.sum(freqWords,axis=1)+alpha*vocabSize)
    theta=np.zeros((2,vocabSize),dtype=np.float64)

    for word in vocabulary:
        theta[0][vocabulary[word]]=np.log((freqWords[0][vocabulary[word]]+alpha))*2-totWords[0]
        theta[1][vocabulary[word]]=np.log((freqWords[1][vocabulary[word]]+alpha))*2-totWords[1]

    # stopwords for wordcloud
    # stopwords=set(stopwords.words('english')
    # cloudPos=WordCloud(background_color='white',max_words=20000,stopwords=stopwords)
    # cloudNeg=WordCloud(background_color='white',max_words=20000,stopwords=stopwords)

    # generate the word cloud
    # cloudPos.generate(" ".join(posWords)).to_image().show()
    # cloudNeg.generate(" ".join(negWords)).to_image().show()

    # # display the Positive Class word cloud
    # plt.imshow(cloudPos,interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    # # display the Negative Class word cloud
    # plt.imshow(cloudNeg,interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    return phi,theta,freqWords


def testConfuMat(posClass,negClass,phi,theta,doStem=False,isBigram=False,isTriGram=False):
    global vocabSize,vocabulary

    # postive review and negative review file
    filePos=glob.glob(posClass+"/*.txt")
    fileNeg=glob.glob(negClass+"/*.txt")

    results=np.zeros([2,2],dtype=np.int_) # [0] for correct result and [1] for wrong result

    # traversing positive review file
    for f in filePos:
        likelihood=np.copy(phi) # likelihood denotes likelihood erview being positive and negative
        lines=splitToWords(f,doStem,isBigram,isTriGram) # spilling the words and removing punctuations

        for i in range(0,len(lines)):
            if lines[i] in vocabulary: # checking weather a word exist or not in vocabulary
                likelihood[0]+=theta[0][vocabulary[lines[i]]]
                likelihood[1]+=theta[1][vocabulary[lines[i]]]

        if likelihood[0]>=likelihood[1]:
            results[0][0]+=1
        else:
            results[1][0]+=1

    # traversing negative review file
    for f in fileNeg:
        likelihood=np.copy(phi) # likelihood denotes likelihood erview being positive and negative
        lines=splitToWords(f,doStem,isBigram,isTriGram) # spilling the words and removing punctuations

        for i in range(0,len(lines)):
            if lines[i] in vocabulary: # checking weather a word exist or not invocabulary
                likelihood[0]+=theta[0][vocabulary[lines[i]]]
                likelihood[1]+=theta[1][vocabulary[lines[i]]]

        if likelihood[1]>=likelihood[0]:
            results[1][1]+=1
        else:
            results[0][1]+=1

    return results


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


def findScores(confusionMat):

    precision=confusionMat[0][0]/(np.sum(confusionMat[0,:]))
    recall=confusionMat[0][0]/(np.sum(confusionMat[:,0]))
    f1Score=2*(precision*recall)/(precision+recall)

    return precision,recall,f1Score


def splitToWords(fileName,doStem=False,isBigram=False,isTriGram=False):
    with open(fileName,'r',encoding="latin-1") as f:
        lines=f.readlines() # Get lines as a list of strings
        # if doStem:
        #     lines=list(map(str.strip, lines)) # Remove /n characters
        #     lines=list(filter(None, lines)) # Remove empty strings
        wordList=[] # List of all Words
        for i in range(0,len(lines)):
            lines[i]=lines[i].lower() # making all words lowercase
            lst=[]

            if doStem:
                StopWords = set(stopwords.words("english"))
                # lines[i]=lines[i].translate(lines[i].maketrans('', '', string.punctuation)) # removing punctutations
                lst=[stemmer.stem(word) for word in lines[i].split() if word not in StopWords]
            else:
                lst=lines[i].split()
            if isBigram:
                lstLen=len(lst)
                for i in range(1,lstLen):
                    lst.append(lst[i-1]+" "+lst[i])
            if isTriGram:
                lstLen=len(lst)
                for i in range(2,lstLen):
                    lst.append(lst[i-2]+" "+lst[i-1]+" "+lst[i])
            wordList.extend(lst)
    return wordList


def main():
    global vocabSize,vocabulary

    trainingDir=sys.argv[1]
    testDir=sys.argv[2]

    results5=pickle.load(open("../results5.pickle","rb"))

    print("\nModel Training Started with BiGrams without Stemming")
    phi,theta,_=train(trainingDir+"/pos",trainingDir+"/neg",False,True)
    print("Model Training Completed")

    results12=testConfuMat(trainingDir+"/pos",trainingDir+"/neg",phi,theta,False,True)
    results11=testConfuMat(testDir+"/pos",testDir+"/neg",phi,theta,False,True)

    print("\nAccuracy over Training Data after using BiGrams without Stemming: ",findAcc(results12),"\n")
    print("Accuracy over Test Data after using BiGrams without Stemming: ",findAcc(results11),"\n")
    print("\nAccuracy increased by ",findAcc(results11)-findAcc(results5)," percent after using Bigrams without stemming")

    print("\nModel Training Started with BiGrams over Stemming")
    phi,theta,_=train(trainingDir+"/pos",trainingDir+"/neg",True,True)
    print("Model Training Completed")

    results9=testConfuMat(trainingDir+"/pos",trainingDir+"/neg",phi,theta,True,True)
    results6=testConfuMat(testDir+"/pos",testDir+"/neg",phi,theta,True,True)

    print("\nAccuracy over Training Data after using BiGrams over Stemming: ",findAcc(results9),"\n")
    print("Accuracy over Test Data after using BiGrams over Stemming: ",findAcc(results6),"\n")
    print("\nAccuracy increased by ",findAcc(results6)-findAcc(results5)," percent after using Bigrams over stemming")

    print("\nModel Training Started with TriGrams over BiGrams over Stemming")
    vocabSize=0
    vocabulary.clear()
    phi,theta,_=train(trainingDir+"/pos",trainingDir+"/neg",True,True,True)
    print("Model Training Completed")

    results10=testConfuMat(trainingDir+"/pos",trainingDir+"/neg",phi,theta,True,True,True)
    results7=testConfuMat(testDir+"/pos",testDir+"/neg",phi,theta,True,True,True)
    print("\nAccuracy over Training Data after using TriGrams over BiGrams over Stemming: ",findAcc(results10),"\n")
    print("Accuracy over Test Data after using TriGrams over BiGrams over Stemming: ",findAcc(results7),"\n")
    print("\nAccuracy increased by ",findAcc(results7)-findAcc(results6)," percent after using TriGrams over Bigrams")

    os.chdir("..")
    file6="results6.pickle"
    fileobj6=open(file6,"wb")
    pickle.dump(results6,fileobj6)
    fileobj6.close()

    file7="results7.pickle"
    fileobj7=open(file7,"wb")
    pickle.dump(results7,fileobj7)
    fileobj7.close()

    results2=pickle.load(open("results2.pickle","rb"))

    print("\nParameters of performance over First learned Model")
    precision2,recall2,f1Score2=findScores(results2)
    print("Precision: ",precision2)
    print("Recall: ",recall2)
    print("F1 Score: ",f1Score2)

    print("\nParameters of performance over Stemmed Model")
    precision5,recall5,f1Score5=findScores(results5)
    print("Precision: ",precision5)
    print("Recall: ",recall5)
    print("F1 Score: ",f1Score5)

    print("\nParameters of performance over BiGram Model")
    precision6,recall6,f1Score6=findScores(results6)
    print("Precision: ",precision6)
    print("Recall: ",recall6)
    print("F1 Score: ",f1Score6,"\n")

main()
