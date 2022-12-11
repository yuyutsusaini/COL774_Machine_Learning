import pickle
import os
import sys

def main():

    trainingDir=sys.argv[1]
    testDir=sys.argv[2]


    # os.chdir("..")
    # file5="results5.pickle"
    # fileobj5=open(file5,"wb")
    # pickle.dump(results5,fileobj5)
    # fileobj5.close()

    os.chdir("..")
    confusionMatrixA=pickle.load(open("confusionMatrixA.pickle","rb"))

    print("\nThe Confusion Matrix for Part A is:")
    print(confusionMatrixA)

    confusionMatrixB=pickle.load(open("confusionMatrixB.pickle","rb"))
    
    print("\nThe Confusion Matrix for Part B is:")
    print(confusionMatrixB)

main()