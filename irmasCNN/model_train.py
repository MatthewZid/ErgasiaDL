# -*- coding: utf-8 -*-
from sklearn import model_selection
from torch import optim
from torch.utils.data import DataLoader ,random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import cv2
import os
import copy
import seaborn as sn
import time
import sys
import signal
import cv2
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
 
torch.autograd.set_detect_anomaly(True)

savemodel = True
show_plots = True
writestats = True
earlystop = True
show_plots_sigint = False
use_scheduler = True

train_losses = []
val_losses = []
train_f1 = []
val_f1 = []

train_csv = r'../csv/train_mel_stereo_dataset.csv'
val_csv = r'../csv/val_mel_stereo_dataset.csv'
train_img_dir = r'../meldata_stereo_masked/train'
val_img_dir = r'../meldata_stereo_masked/val'

class irmasCNN(nn.Module):
    def __init__(self):
        super(irmasCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(4,4))
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3))
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2))
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2))

        # nn.init.kaiming_uniform_(self.cnn1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cnn2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cnn3.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cnn4.weight, mode='fan_in', nonlinearity='relu')

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.drop = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(64*6*9, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 11)

        # nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.uniform_(self.fc3.weight, -0.01, 0.01)

    def forward(self, x):
        x = self.drop(self.pool(F.relu(self.bn1(self.cnn1(x)))))
        x = self.drop(self.pool(F.relu(self.bn2(self.cnn2(x)))))
        x = self.drop(self.pool(F.relu(self.bn3(self.cnn3(x)))))
        x = self.drop(self.pool(F.relu(self.bn4(self.cnn4(x)))))
        x = x.view(-1, 64*6*9) # flatten the CNN output
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

class irmasLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, nclasses):
        super(irmasLSTM, self).__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=0.25)

        self.lstm = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True) # (batch, seq, features)
        self.fc = nn.Linear(hidden_size, nclasses)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(device))

        # x = x.view(-1, x.size(-3)*x.size(-2)*x.size(-1))
        print(x.size())

        _, (lstm_hn, _) = self.lstm(x, (h0, c0))
        print(lstm_hn.size())
        lstm_hn = lstm_hn.view(-1, self.hidden_size)
        x = self.fc(self.dropout(F.relu(lstm_hn)))

        return x

#############################################################################
############################################################################
#############################################################################
class MelSpecDataset(Dataset):
  def __init__(self , csv_file , data_dir):
    
    self.df = pd.read_csv(csv_file, header=None)
    self.data_dir = data_dir
    
  def __getitem__(self, idx):
    d = self.df.iloc[idx]
    ch1 = cv2.imread(os.path.join(d[0], d[0].split('/')[-1]+'_ch1.png'))
    ch1 = cv2.cvtColor(ch1, cv2.COLOR_BGR2GRAY)
    ch2 = cv2.imread(os.path.join(d[0], d[0].split('/')[-1]+'_ch2.png'))
    ch2 = cv2.cvtColor(ch2, cv2.COLOR_BGR2GRAY )
    ch1 = cv2.normalize(ch1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ch2 = cv2.normalize(ch2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    arr = np.array([[ch1, ch2]], dtype=np.float32)
    # trans = transforms.ToTensor()
    # arr = trans(ch1)
    arr = torch.from_numpy(arr)
    label = torch.tensor(d[1])
    return arr, label
  
  def __len__(self):
    return len(self.df)
##############################################################################
train = MelSpecDataset(train_csv, train_img_dir)
val = MelSpecDataset(val_csv, val_img_dir)


batch_size = 32

print('Train, val samples: {}, {}'.format(len(train), len(val)))
print('Train dataset: {}'.format(train_csv))
print('Val dataset: {}'.format(val_csv))

dataloader = {'train':DataLoader(train,
                                  shuffle = True ,
                                 batch_size = batch_size, num_workers=4),
              'val': DataLoader(val , 
                                shuffle = True ,
                                batch_size = batch_size, num_workers=4)}


lr = 0.0001
nlayers = 1
input_size = train[0][0].size(2)
hidden_size = 32
n_classes = 11
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"


model = irmasCNN().to(device)
model_summary = str(summary(model, input_size=(batch_size, 2, 124, 166), col_names=["output_size", "num_params", "kernel_size"], device=device, verbose=0))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=5, verbose=True)

def createCheckpoint(epoch, model, optimizer, filename=r'/LatestCheckpoint.pt'):
    checkpoint = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              "batch_size":batch_size,
    } # save all important stuff
    torch.save(checkpoint , filename)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def one_hot(target):
    multis = []
    for tar in target:
        multi = [0]*11
        multi[int(tar)] = 1
        multis.append(multi)
    return torch.tensor(multis)

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:      
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def save_model(epoch, model, optimizer, filename, batch):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'batch': batch
    }

    torch.save(checkpoint, filename)

def load_model(path, model, optim):
    model_name = path.split('/')[-1]
    if os.path.exists(path):
        global batch_size
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        batch_size = checkpoint['batch']
        print("Model \'{}\' loaded\n".format(model_name))
        return epoch+1
    else:
        print("Model \'{}\' doesn't exist...\n".format(model_name))
        return 0

def conf_matr(trues, preds, classes_num, phase, epoch):
    cfmatr = confusion_matrix(trues, preds)
    dfcm = pd.DataFrame(cfmatr, index = [i for i in range(classes_num)],
                    columns = [i for i in range(classes_num)])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(dfcm, annot=True)
    plt.savefig('../stats/{}/{}_confusion_matrix_{:d}.png'.format(phase, phase, epoch))
    plt.close()

def stats(traint, trainp, valt, valp, classes_num, epoch):
    with open('../stats/{}/{}_report_{:d}.txt'.format('train', 'train', epoch), "w") as f:
        f.write(classification_report(traint, trainp, zero_division=0))
    conf_matr(traint, trainp, classes_num, 'train', epoch)

    with open('../stats/{}/{}_report_{:d}.txt'.format('val', 'val', epoch), "w") as f:
        f.write(classification_report(valt, valp, zero_division=0))
    conf_matr(valt, valp, classes_num, 'val', epoch)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = -1
        self.model = None
        self.optimizer = None
        self.batch = -1
        self.ytrues = []
        self.ypreds = []
        self.classes_num = -1
        
    def __call__(self, val_loss, epoch, model, optimizer, batch, ytrues, ypreds, classes_num):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.keepmodel(epoch, model, optimizer, batch, ytrues, ypreds, classes_num)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.keepmodel(epoch, model, optimizer, batch, ytrues, ypreds, classes_num)
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose: print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                stats(self.ytrues[0], self.ypreds[0], self.ytrues[1], self.ypreds[1], self.classes_num, self.epoch)
                if savemodel: save_model(self.epoch, self.model, self.optimizer, r'../models/Checkpoints/Checkpoint_{}_{}.pt'.format(self.epoch+1, time.time()), self.batch)
                self.early_stop = True
    
    def keepmodel(self, epoch, model, optimizer, batch, ytrues, ypreds, classes_num):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.batch = batch
        self.ytrues = ytrues
        self.ypreds = ypreds
        self.classes_num = classes_num

def model_train(model , data_loader , criterion , optimizer, scheduler, early_stopping, num_epochs = 30):
    for epoch in trange(num_epochs, desc = 'Epochs'):
        result = []
        ytrues = [[],[]]
        ypreds = [[],[]]
        for phase in ['train', 'val']:
            if phase == "train":
                model.train()
            else:  
                model.eval()
       
            running_loss = 0.0
            running_corrects = 0.0  
            for data , target in data_loader[phase]:
                data , target = data.to(device)  , target.to(device)
                data = torch.squeeze(data, 1).to(device)
                target = target.type(torch.LongTensor).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)
                    y_true = torch.clone(target).to(device).to(torch.int).numpy()
                    soft = F.softmax(output, dim=1).to(device).to(torch.float32)
                    # loss = criterion(output, one_hot(target).type_as(output))
                    loss = criterion(output, target)
                    preds = torch.argmax(soft, dim = 1).to(device)
                    y_pred = preds.numpy()
                    pos = 0
                    if phase == 'val': pos = 1
                    ytrues[pos].extend(y_true)
                    ypreds[pos].extend(y_pred)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()                
                        #plot_grad_flow(model.named_parameters())
                        optimizer.step()
                running_loss += loss.item() * data.size(0)
                running_corrects += f1_score(target.to(device).to(torch.int).numpy() ,preds.to(device).to(torch.int).numpy() , average="macro") * data.size(0)

    
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_f1.append(epoch_acc)

            else:
                val_losses.append(epoch_loss)
                val_f1.append(epoch_acc)

            result.append('{} Loss: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if earlystop:
                    if not early_stopping.early_stop:
                        early_stopping(epoch_loss, epoch, model, optimizer, batch_size, ytrues, ypreds, n_classes)
            
            if use_scheduler:
                if phase == 'val':
                    scheduler.step(epoch_loss)

        print(result)

        if writestats:
            if epoch + 1 == num_epochs and early_stopping.early_stop == False:
                stats(ytrues[0], ypreds[0], ytrues[1], ypreds[1], n_classes, epoch)

        if savemodel:
            if epoch + 1 == num_epochs and early_stopping.early_stop == False:
                try:
                    save_model(epoch, model, optimizer, r'../models/Checkpoints/Checkpoint_{}_{}.pt'.format(epoch+1, time.time()), batch_size)
                except:
                    print('Error: Model could not be saved!')

    return model



early_stopping = EarlyStopping(patience=7)

cnn = model_train(model, dataloader, criterion, optimizer, lr_scheduler, early_stopping, num_epochs = 100)

with open('../stats/model_summary.txt', 'w') as f:
    f.write(model_summary)

if show_plots:
    epoch_space = np.arange(1, len(val_losses)+1)
    ymin = min(min(train_losses), min(val_losses))
    ymax = max(max(train_losses), max(val_losses))
    legendlist = []
    plt.figure()
    plt.plot(epoch_space, train_losses)
    plt.plot(epoch_space, val_losses)
    if early_stopping.early_stop:
        plt.vlines(float(early_stopping.epoch+1), ymin, ymax, colors='r', linestyles='dashed')
        legendlist.extend(["train loss", "validation loss", "early stopping checkpoint"])
    else: legendlist.extend(["train loss", "validation loss"])
    plt.legend(legendlist, loc="upper right")
    plt.savefig('../stats/loss.png')
    plt.close()

    ymin = min(min(train_f1), min(val_f1))
    ymax = max(max(train_f1), max(val_f1))
    legendlist.clear()
    legendlist = []
    plt.figure()
    plt.plot(epoch_space, train_f1)
    plt.plot(epoch_space, val_f1)
    if early_stopping.early_stop:
        plt.vlines(float(early_stopping.epoch+1), ymin, ymax, colors='r', linestyles='dashed')
        legendlist.extend(["train f1", "validation f1", "early stopping checkpoint"])
    else: legendlist.extend(["train f1", "validation f1"])
    plt.legend(legendlist, loc="lower right")
    plt.savefig('../stats/f1.png')
    plt.close() 
