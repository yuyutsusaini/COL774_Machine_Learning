!pip install transformers

import numpy as np 
import pandas as pd 
import os
import re
import torch
from os.path import join
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
import os
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
from transformers import BertTokenizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
import csv
from os import sys

input_dir = sys.argv[1]

device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



def clean_text(text):

    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)

    text = text.replace('#',' ')
    text = text.replace('@',' ')
    symbols = re.compile(r'[^A-Za-z0-9 ]')
    text = symbols.sub(r'',text)
    
    text = re.sub(r'\d+', ' ', text)
    
    text = text.lower()
    
    list2 = text.split(' ')
    list1 =[porter.stem(w) for w in list2 if not w in stop_words]
    text = " ".join(list1)
    return text


def train_loader(dir,tokenizer,batch_size):
    train_x = pd.read_csv(join(dir, 'train_x.csv'))
    train_y = pd.read_csv(join(dir, 'train_y.csv'))
    sentences = train_x.Title.values
    labels = train_y.Genre.values
    for i in range(len(sentences)):
        sentences[i] = clean_text(sentences[i])
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True, 
                            max_length = 80,          
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',  
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset), batch_size = batch_size )
    return train_dataloader

batch_size_train = 16
train_dataloader = train_loader(input_dir,tokenizer,batch_size_train)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 30,output_attentions = False,output_hidden_states = False,)
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8
                )


epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


def train(train_dataloader,epochs,model,device,scheduler,optimizer):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    for epoch in range(0, epochs):
        print('Epochs number' + str(epoch+1) )
        net_loss = 0
        model.train()
        for iter, batch in enumerate(train_dataloader):
            if iter % 40 == 0 and not iter == 0: 
                print('  Batch {:>5,}  of  {:>5,}.'.format(iter, len(train_dataloader)))
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)      
            #In order to clear previoud gradients
            model.zero_grad()        
            out = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)
#             print(type(out[0]))
            net_loss+= out[0].item()
            print(out[0].item())
            out[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            # θ = θ - α(∂L/∂θ)
            optimizer.step()
        avg_loss = net_loss / len(train_dataloader)            
        print(" Average training loss: "+ str(avg_loss))

train(train_dataloader,epochs,model,device,scheduler,optimizer)

def load_test_loader(path,tokenizer,batch_size):
    test_x = pd.read_csv(path )
    sentences = test_x.Title.values
#     for i in range(len(sentences)):
#         sentences[i] = clean_text(sentences[i])

    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 80,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    dataset = TensorDataset(input_ids, attention_masks)

    test_dataloader = DataLoader(
            dataset, 
            batch_size = batch_size
        )
    return test_dataloader


test_batch_size=6
test_dataloader = load_test_loader(join(input_dir,'comp_test_x.csv'),tokenizer,test_batch_size)


def test(model,test_dataloader,device):
    model.eval()
    pred =[]
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():        
            logits = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
        rounded_preds = torch.argmax(logits[0], dim=1)
        for x in rounded_preds:
            pred.append(x.item())
    return pred

pred = test(model,test_dataloader,device)

fields =['Id','Genre']
rows = [[i,pred[i]] for i in range(len(pred)) ]
filename = "comp_test_pred_y.csv"
with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    csvwriter.writerows(rows)