import os
import numpy as np
import subprocess

# # # Q1.1
# train = "Q1/dataset1/train.csv"
# val = "Q1/dataset1/val.csv"
# test = "Q1/dataset1/test.csv"
# outfile = "Q1/output1"
# for part in ['all']:
#     subprocess.run(['python','dt_mammography.py',train,val, test,outfile,part])

# Q1.2
# train = "Q1/dataset2/DrugsComTrain.csv"
# val = "Q1/dataset2/DrugsComVal.csv"
# test = "Q1/dataset2/DrugsComTest.csv"
# outfile = "Q1/output2"
# for part in ['a','b']:
#     subprocess.run(['python','dt_drug_review.py',train,val, test,outfile,part])

# Q2
train = "Q2/dataset/fmnist_train.csv"
test = "Q2/dataset/fmnist_test.csv"
outfile = "Q2/output"
for part in ['b']:
    subprocess.run(['python','neural.py',train,test,outfile,part])