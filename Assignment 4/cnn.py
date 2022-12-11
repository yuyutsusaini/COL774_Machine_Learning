from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

import math
import time
import tqdm
import sys

plt.ion()   # interactive mode
from PIL import Image
import torch.nn.functional as F

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def read_image(train_x,st,batch_size):
    X =[]
    Y= []
    convert_tensor = transforms.ToTensor()
    for i in range(st,st+batch_size):
        img = Image.open(os.path.join(sys.argv[1]+'/images/images/',train_x.iloc[i,1])).convert('RGB')
        fal = convert_tensor(img)
        X.append(fal)
    return X


class Net(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*24*24, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN:

    def __init__(self,train_x,train_y,criterion=nn.CrossEntropyLoss(),num_classes=30,batch_size=100,learning_rate = 0.001,num_epochs = 10):
        self.train_x=train_x
        self.train_y=train_y
        self.criterion=criterion
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.model=Net(self.num_classes)

    def CNN_Train(self):
        self.model = self.model.to(device)
        # # Set Loss function with criterion
        # criterion = nn.CrossEntropyLoss()
        # # Set optimizer with optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay = 0.005, momentum = 0.9)  

        total_step = len(self.train_x)

        for epoch in tqdm.tqdm(range(self.num_epochs)):
            for i in range(0,len(self.train_x),self.batch_size):
                #get data
                
                batch_x = read_image(self.train_x,i,self.batch_size)
                batch_y = self.train_y.iloc[i:i+self.batch_size,1]
                batch_x = torch.stack(batch_x)
                batch_y = torch.tensor(batch_y.tolist())
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(i,len(self.train_x))
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))

        return self.model


    def CNN_Test(self,test_x):
        ids=test_x.iloc[:,0].to_numpy()
        ls=[]
        with torch.no_grad():
            for i in tqdm.tqdm(range(0,len(test_x),self.batch_size)):
                #get data
                test_batch_x = torch.empty(1)
                if i+100>len(test_x):
                    test_batch_x = read_image(test_x,i,self.len(test_x)%100)
                else:
                    test_batch_x = read_image(test_x,i,self.batch_size)
                test_batch_x = read_image(test_x,i,self.batch_size)
                test_batch_x = torch.stack(test_batch_x)
                test_batch_x = test_batch_x.to(device)
                # Forward pass
                outputs = self.model(test_batch_x)
                _, predicted = torch.max(outputs.data, 1)
                ls+=predicted.tolist()

        pred=np.vstack((ids, np.asarray(ls))).T
        return pred

    def CNN_Acc(self,test_x,test_y):
        confusion_matrix = np.zeros((self.num_classes,self.num_classes))
        with torch.no_grad():
            for i in tqdm.tqdm(range(0,len(test_x),self.batch_size)):
                #get data
                test_batch_x = torch.empty(1)
                test_batch_y = torch.empty(1)
                if i+100>len(test_x):
                    test_batch_x = read_image(test_x,i,len(test_x)%100)
                    test_batch_y = test_y.iloc[i:i+len(test_x)%100,1]
                else:
                    test_batch_x = read_image(test_x,i,self.batch_size)
                    test_batch_y = test_y.iloc[i:i+self.batch_size,1]
                # test_batch_x = read_image(test_x,i,self.batch_size)
                # test_batch_y = test_y.iloc[i:i+self.batch_size,1]
                test_batch_x = torch.stack(test_batch_x)
                test_batch_y = torch.tensor(test_batch_y.tolist())
                test_batch_x = test_batch_x.to(device)
                test_batch_y = test_batch_y.to(device)
                # Forward pass
                outputs = self.model(test_batch_x)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(len(predicted)):
                    confusion_matrix[predicted[i]][test_batch_y[i]]+=1

        return confusion_matrix

def main():

    train_y = pd.read_csv(sys.argv[1]+"/train_y.csv")
    train_x = pd.read_csv(sys.argv[1]+"/train_x.csv")
    # print(train_x)
    test_y = pd.read_csv(sys.argv[1]+"/non_comp_test_y.csv")
    test_x = pd.read_csv(sys.argv[1]+"/non_comp_test_x.csv")
    comp_test = pd.read_csv(sys.argv[1]+"/comp_test_x.csv")
    # print(len(train_x))
    # print(len(train_y))
    ran = np.random.permutation(len(train_x))
    # print(ran)

    train_x = train_x.iloc[ran]
    train_y = train_y.iloc[ran]

    cnn_obj = CNN(train_x,train_y)
    start=time.time()
    cnn_obj.CNN_Train()
    end=time.time()
    con_mat=cnn_obj.CNN_Acc(test_x,test_y)
    print("Time taken :", end-start)
    print("Confusion Matrix over test :" , con_mat)
    print("Accuracy over the test set :", np.trace(con_mat)/np.sum(con_mat))
    prediction = cnn_obj.CNN_Test(test_x)
    prediction=prediction.astype(int)
    df = pd.DataFrame(prediction, columns=['Id', 'Genre'])
    df.to_csv('non comp test pred y.csv', mode='a', index=False, header=True)
    # CNN_Test(model,test_x,test_y)

main()
