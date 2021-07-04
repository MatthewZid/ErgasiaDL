# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from sklearn.metrics import f1_score
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
import gc
from MelSpecDataset import MelSpecDataset
from sklearn.metrics import confusion_matrix, classification_report


labels = 'cel cla flu gac gel org pia sax tru vio voi'.split()
csv = r'C:\Datasets\Irmas\128x128.csv'
data_dir = r'C:\Datasets\Irmas\train_128x128'
checkpoint_dir = r'C:\Datasets\Irmas\checkpoints'

name = r'C:\Datasets\Irmas\GrayVGG16_FC_BN_epoch100_batchsize32.pth'


model = torch.load(name,  map_location=torch.device('cpu')).double()
model.classifier = nn.Sequential(nn.Linear(in_features = 8192, out_features = 4096),
                              nn.LeakyReLU(),
                              nn.BatchNorm1d(4096),
                              nn.Dropout(0.4),
                              nn.Linear(in_features = 4096, out_features = 512),
                              nn.LeakyReLU(),
                              nn.BatchNorm1d(512),
                              nn.Dropout(0.2),
                              nn.Linear(in_features = 512, out_features = 11)).double()

def createCheckpoint(score, f1):
    filename = os.path.join(checkpoint_dir, f'epoch{total_epochs}_{f1}.pt')
    checkpoint = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              "batch_size":batch_size,
    } 
    torch.save(checkpoint , filename)
    
    

def one_hot(target):
    multis = []
    for tar in target:
        multi = [0]*11
        multi[tar] = 1
        multis.append(multi)
    return torch.tensor(multis)

train_losses = []
val_losses = []
train_scores = []
val_scores = []
total_epochs = 0


df = pd.read_csv(csv)

        
train_df, test_df = train_test_split(df, test_size = 0.1, random_state = 13)
train_df, val_df = train_test_split(train_df, test_size = 0.15, random_state = 13)

train = MelSpecDataset(train_df, data_dir)
val = MelSpecDataset(val_df, data_dir)


batch_size = 32
batches = round((len(train) + len(val))/32)

dataloader = {"train":DataLoader(train,
                                  shuffle = True ,
                                 batch_size = batch_size),
              "val": DataLoader(val , 
                                shuffle = True ,
                                batch_size = batch_size)}
print('Total train, validation instances: {}, {}'.format(len(train), len(val)))

lr = 0.00005
device = torch.device('cpu')#('cuda:0' if torch.cuda.is_available() else 'cpu')

#model = CNN()
ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.02)
criterion = nn.BCEWithLogitsLoss()



'''
model.load_state_dict(torch.load(r'C:\Datasets\Irmas\checkpoints\Try_3\128x128_epoch18_0.48093387551390915.pt')['model_state_dict'])
optimizer.load_state_dict(torch.load(r'C:\Datasets\Irmas\checkpoints\Try_3\128x128_epoch18_0.48093387551390915.pt')['optimizer_state_dict'])

'''                                                  

num_epochs = 3
for epoch in trange(num_epochs, desc = 'Epochs'):
    batch = 1
    total_epochs += 1
    if epoch == 0:
        print('Epoch {}'.format(epoch))
    result = []
    for phase in ['train', 'val']:
        if epoch == 0:
            print('Phase {}'.format(phase))
        if phase == 'train':
            model.train()
        else:  
            model.eval()
  
        running_loss = 0.0
        running_corrects = 0.0  
        load_time = time.time()
        print("Loading..")
        for im , target in dataloader[phase]:
            load_end = time.time()
            print("loading took {:.4f} mins.".format((load_end-load_time)/60))
            print('New batch {}/{}'.format(batch, batches))
            batch += 1
            batch_start = time.time()
            im, target = im.to(device).double(), target.to(device)
            with torch.set_grad_enabled(phase == 'train'):
                output = model(im)
                loss = criterion((output), one_hot(target).type_as(output))
                preds = one_hot(torch.argmax(output, dim = 1))
                if phase == 'train' :
                    optimizer.zero_grad()
                    loss.backward()                
                    optimizer.step()
            batch_end = time.time()
            batch_f1 = f1_score(one_hot(target).to('cpu').to(torch.int).numpy() 
                                         ,preds.to('cpu').to(torch.int).numpy() 
                                         ,average = 'macro',
                                         labels = np.unique(preds))
            print('Batch time: {:.4f} mins, f1 : {:.4f}'.format((batch_end-batch_start)/60, batch_f1))
            running_loss += loss.item() * im.size(0)
            running_corrects += f1_score(one_hot(target).to('cpu').to(torch.int).numpy() 
                                         ,preds.to('cpu').to(torch.int).numpy() 
                                         ,average = 'macro',
                                         labels = np.unique(preds)) * im.size(0)
            gc.collect()
            load_time = time.time()
            print("Loading..")   
        epoch_loss = running_loss / len(dataloader[phase].dataset)
        epoch_acc = running_corrects / len(dataloader[phase].dataset)
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_scores.append(epoch_acc)
        else:
            val_losses.append(epoch_loss)
            val_scores.append(epoch_acc)
        result.append('{} Loss: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    print(result)
    try:
        createCheckpoint(total_epochs, epoch_acc)
    except:
        print('Saving failed!')




epoches = np.arange(1, len(val_losses)+1)
plt.plot(epoches, train_losses)
plt.plot(epoches, val_losses)
plt.xlim(1, 9)
plt.legend(["train loss", "validation loss"], loc ="lower left")
plt.show()

plt.plot(epoches, train_scores)
plt.plot(epoches, val_scores)
plt.xlim(1, 9)
plt.legend(["train f1", "validation f1"], loc ="lower left")
plt.show()


test = MelSpecDataset(test_df, data_dir)
model.eval()
testloader = DataLoader(test, shuffle = True, batch_size = 32)

y_true, y_pred = [], []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)   
        for label in labels:
          #  print(label)
            y_true.append(label.data)
            
        for output in outputs:
            pred = torch.argmax(output)
            y_pred.append(pred.data)
            
conf=confusion_matrix(y_true,y_pred)
plt.matshow(conf)
rep=classification_report(y_true,y_pred)
f1=f1_score(y_true,y_pred,average = 'macro')




