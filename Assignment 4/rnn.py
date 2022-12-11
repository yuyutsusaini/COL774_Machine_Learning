import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from os import sys
from torchtext import data,vocab
import csv 
from os.path import join
import warnings
warnings.filterwarnings('ignore')
# pd.set_option('display.max_colwidth', -1)
# import re

glove = {}
def load_glove(input_folder):
    file = open(join(input_folder,"glove.6B.300d.txt"), "r")
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove[word] = vector
    file.close()


def getAcc(pred,actual):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            correct += 1
    return correct/len(pred)


def pad_features(reviews_int, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review 
        
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

class BiRNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim,hidden_dim2,
                 output_dim, n_layers, bidirectional=True, dropout=0.5):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, 
                           bidirectional = bidirectional, dropout=dropout, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim, n_layers, dropout = dropout, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,128)
        self.fc2 = nn.Linear(128, output_dim)        
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        self.activation = nn.Tanh()
        self.sof = nn.Softmax()
        self.lr = nn.LeakyReLU()
        
    def forward(self, text):
        
        embedded = self.dropout(self.embedding(text))
        
        output, hidden = self.rnn(embedded)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        hidden = self.activation(self.fc1(hidden))
        hidden = self.fc2(hidden)

        return self.sof(hidden)

def obtain_x(train,test):
    get_index = {}
    glove_dict = {}
    train_X_list = []
    ind = 1
    embed_keys = glove.keys()
    for x in train['Title']:
            tokens = x.split(' ')
            new_list = []
            for i in tokens:
                if((i in embed_keys)  and (i not in get_index.keys())):
                    glove_dict[ind] = glove[i]
                    get_index[i] = ind
                    new_list.append(ind)
                    ind=ind+1   

                elif(i not in get_index.keys()):
                    glove_dict[ind] = np.random.normal(scale=0.7, size=(300, )).astype(np.float)
                    get_index[i] = ind
                    new_list.append(ind)
                    ind=ind+1   

                else:
                    new_list.append(get_index[i])
            train_X_list.append(new_list)
    test_X_list = []
    ind = len(get_index)+1

    embed_keys = glove.keys()
    for x in test['Title']:
            tokens = x.split(' ')
            new_list = []
            for i in tokens:
                if((i in embed_keys)  and (i not in get_index.keys())):
                    glove_dict[ind] = glove[i]
                    get_index[i] = ind
                    new_list.append(ind)
                    ind=ind+1   

                elif(i not in get_index.keys()):
                    glove_dict[ind] = np.random.normal(scale=0.7, size=(300, )).astype(np.float)
                    get_index[i] = ind
                    new_list.append(ind)
                    ind=ind+1   

                else:
                    new_list.append(get_index[i])
            test_X_list.append(new_list)
    return train_X_list,test_X_list,glove_dict

def load_data(train_x,train_y,test_x,batch_size=100):
    train = pd.concat([train_x, train_y["Genre"]], axis=1, join="inner")
#     train.head()
    train.drop(['Cover_image_name'],axis =1, inplace = True)
    train.drop(['Id'],axis=1, inplace = True)
#     test_x.head()

    test_x.drop(['Cover_image_name'],axis =1, inplace = True)

    train_X_list, test_X_list, glove_dict = obtain_x(train,test_x)
    train_X_list = pad_features(train_X_list,56)
    train_y_list=[]

    for i in train['Genre']:
        temp = [float(0) for _ in range(30)]
        temp[i]=float(1)
        train_y_list.append(np.array(temp).astype(np.float32))
    train_y_list=np.array(train_y_list)
    
    test_X_list = pad_features(test_X_list,56)
    test_id_list=[]
    for i in test_x['Id']:
        test_id_list.append(i)
    test_id_list=np.array(test_id_list)
    
    glove_dict[0] = np.array(np.zeros(300)).astype(np.float)
    train_data = TensorDataset(torch.from_numpy(train_X_list),torch.from_numpy(train_y_list))
    train_loader = DataLoader(train_data, batch_size = batch_size, drop_last = True,shuffle = False)
    test_data = TensorDataset(torch.from_numpy(test_X_list),torch.from_numpy(test_id_list))
    test_loader = DataLoader(test_data, batch_size = batch_size, drop_last = True,shuffle = False)
    return train_loader, test_loader, glove_dict



def train(train_loader,net, optimizer, criterion,epochs,is_gpu):
    clip=5 
    net = net.float()
    if(is_gpu):
        net.cuda()
    epoch_loss = 0
    epoch_acc = 0
    counter = 0
    for e in range(epochs):
        net.train()  
        for inputs, labels in train_loader:
            inputs = inputs.type(torch.LongTensor)
            if(is_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            counter += 1
            net.zero_grad()
            predictions = net(inputs)
            loss = criterion(predictions, labels)
            rounded_preds = torch.argmax(predictions, dim=1)
            correct = (rounded_preds == torch.argmax(labels, dim=1)).float() 

            acc = correct.sum()/len(correct)

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if counter % 100 == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(epoch_loss/counter)
                      ,"Acc: {:.6f} %".format((epoch_acc/counter)*100))
            
    return net

def test(test_loader,net,is_gpu):
    pred = []

    net.eval()
    # iterate over test data
    for inputs,idp in test_loader:
        if(is_gpu):
            inputs = inputs.cuda()
        output = net(inputs)
        rounded_preds = torch.argmax(output, dim=1)

        if(is_gpu):
            rounded_preds = rounded_preds.cuda()
        l = len(pred)
        for i in rounded_preds:
            pred.append(i.item())
    return pred


def main():

    input_folder = sys.argv[1]
    train_y = pd.read_csv(join(input_folder, "train_y.csv"))
    train_x = pd.read_csv(join(input_folder, "train_x.csv"))
    test_x = pd.read_csv(join(input_folder, "non_comp_test_x.csv"))
    load_glove(input_folder)
    train_loader, test_loader,new_embedding_index = load_data(train_x.copy(),train_y,test_x.copy())

    output_size = 30
    hidden_dim = 128
    hidden_dim2 = 128
    n_layers = 1
    embedding_dim = 300
    vocab_size = len(new_embedding_index)
    net = BiRNN(vocab_size, embedding_dim, hidden_dim, hidden_dim2,  output_size,n_layers)
    train_on_gpu = torch.cuda.is_available()
    # training params
    epochs =1
    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    #train
    net = train(train_loader,net, optimizer, criterion,epochs,train_on_gpu)

    #test
    pred = test(test_loader,net,train_on_gpu)
    # print(getAcc(pred,labels_y)*100)
    fields =['Id','Genre']
    rows = [[i,pred[i]] for i in range(len(pred)) ]
    filename = "non_comp_test_pred_y.csv"
    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)

main()