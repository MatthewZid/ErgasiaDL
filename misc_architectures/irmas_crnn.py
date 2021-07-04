# -*- coding: utf-8 -*-
from sklearn import model_selection
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

train_losses = []
val_losses = []
train_f1 = []
val_f1 = []

########################### System setup ###########################
def handler(signal_received, frame):
    print("\n[System_{}]: SIGINT received".format(time.time()))
    if show_plots_sigint:
        epoch_space = np.arange(1, len(val_losses)+1)
        plt.figure()
        plt.plot(epoch_space, train_losses)
        plt.plot(epoch_space, val_losses)
        plt.legend(["train loss", "validation loss"], loc="lower left")
        plt.savefig('../stats/loss.png')
        plt.close()

        plt.figure()
        plt.plot(epoch_space, train_f1)
        plt.plot(epoch_space, val_f1)
        plt.legend(["train f1", "validation f1"], loc="lower left")
        plt.savefig('../stats/f1.png')
        plt.close()
        print('Plots saved!')
    print("[System_{}]: Exiting...".format(time.time()))
    sys.exit(0)

signal.signal(signal.SIGINT, handler)
####################################################################

labels = 'cel cla flu gac gel org pia sax tru vio voi'.split()
train_csv = r'../train_mel_dataset_mono.csv'
val_csv = r'../val_mel_dataset_mono.csv'
test_csv = r'../test_mel_dataset_mono.csv'
img_dir = r'../meldata_mono'
# meta = pd.read_csv(csv)
# ##############################################################################
# to_plot = meta.iloc[:,1].apply(lambda x: np.array(eval(x)), 0)
# to_plot = pd.DataFrame(meta.one_hot.apply(lambda x: np.array(eval(x)), 0).tolist())
# to_plot.columns = labels
# fig1, ax1 = plt.subplots()
# to_plot.sum(axis=0).plot.pie(autopct = '%1.1f%%', y = labels,
#                                    shadow = True,
#                                    startangle = 90,
#                                    ax = ax1)
# plt.title('Training class percentages')
# ax1.axis('equal')
# plt.show()
###############################################################################
#
#examples = [0]*len(labels)
#for i, row in meta.iterrows():  
#   label = row[1:].tolist()
#    try:
#       target = label.index(1)
#    except:
#        continue
#    if examples[target] == 1:
#        continue
#    else:
#        path = os.path.join(r'C:\Users\user\Music\IRMAS-TrainingData\train',
#                            labels[target])
#        example = librosa.read(os.listdir(path)[13])
#        examples[target] = (example, row.image)
#    if 0 not in examples:
#        break
##################


# class irmasCNN(nn.Module):
#     def __init__(self):
#         super(irmasCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))     
#         self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (0, 0),
#                                dilation = (1, 1))
#         self.conv7 = nn.Conv2d(in_channels = 128, out_channels = 256, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 256, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)

        
#         self.pool = nn.MaxPool2d(kernel_size = 2, 
#                                  stride = 1)
#         self.drop = nn.Dropout(p = 0.25)
#         self.drop2 = nn.Dropout(p = 0.5)
#         self.act = nn.LeakyReLU(negative_slope = 0.3)
#         self.sigmoid = nn.Sigmoid()
#         self.fc = nn.Linear(512*15*15, 11)
    

#     def forward(self, x):
#         x = self.drop(self.act(self.pool(self.conv2(self.conv1(x)))))
#         x = self.drop(self.act(self.pool(self.conv4(self.conv3(x)))))
#         x = self.drop(self.act(self.pool(self.conv6(self.conv5(x)))))
#         x = self.act(self.pool((self.conv8(self.conv7(x)))))
#         x = F.max_pool2d(x, kernel_size = x.size()[2:])
#         print(x.shape)
#         x = x.view(-1, 512*15*15) 
#         x = self.drop2(self.fc(x))
#         x = self.sigmoid(x)
#         return x

class irmasCNN(nn.Module):
    def __init__(self):
        super(irmasCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(4,4))
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3))
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2))
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2))

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.drop = nn.Dropout2d(p=0.25)

        self.fc1 = nn.Linear(64*6*9, 200)
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.cnn1(x))))
        x = self.pool(F.relu(self.bn2(self.cnn2(x))))
        x = self.pool(F.relu(self.bn3(self.cnn3(x))))
        x = self.pool(F.relu(self.bn4(self.cnn4(x))))
        x = x.view(-1, 64*6*9)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

class irmasEncoder(nn.Module):
    def __init__(self):
        super(irmasEncoder, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5))
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4,4)) # 16*33*3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2)) # 32*16*1

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.drop = nn.Dropout2d(p=0.25)

    def forward(self, x):
        outlist = []

        for t in range(x.size(1)):
            inp = x[:, t, :, :, :]
            out = self.pool(F.relu(self.bn1(self.cnn1(inp))))
            out = self.pool(F.relu(self.bn2(self.cnn2(out))))
            out = self.pool(F.relu(self.bn3(self.cnn3(out))))
            outlist.append(out)
        
        outlist = torch.stack(outlist, dim=0).transpose_(0, 1)

        return outlist

class irmasDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, nclasses):
        super(irmasDecoder, self).__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=0.25)

        self.lstm = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True) # (batch, seq, features)
        self.fc = nn.Linear(hidden_size, nclasses)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(device))
        x = x.view(x.size(0), x.size(1), x.size(-3)*x.size(-2)*x.size(-1))

        output, (_, _) = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        x = self.fc(self.dropout(F.relu(output)))

        return x

def initialize_weights_enc(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def initialize_weights_dec(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

# class irmasCNN(nn.Module):
#     def __init__(self):
#         super(irmasCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, 
#                                kernel_size = (2, 2), 
#                                stride = (2, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, 
#                                kernel_size = (2, 2), 
#                                stride = (2, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, 
#                                kernel_size = (2, 2), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))     
#         self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, 
#                                kernel_size = (2, 2), 
#                                stride = (1, 1),
#                                padding = (1, 1),
#                                dilation = (1, 1))
#         self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512, 
#                                kernel_size = (3, 3), 
#                                stride = (1, 1),
#                                padding = (0, 0),
#                                dilation = (1, 1))
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn5 = nn.BatchNorm2d(512)

        
#         self.pool = nn.MaxPool2d(kernel_size = 3, 
#                                  stride = 2)
#         self.drop = nn.Dropout(p = 0.5)
#         self.fcdrop = nn.Dropout(p = 0.5)
#        # self.softmax = nn.Softmax()
#         self.fc = nn.Linear(int(64*2*2), int(round(64*2*2)/2))
#         self.fc1 = nn.Linear(int(round(64*2*2)/2), 11)

#     def forward(self, x):
#         x = self.drop(self.pool(F.relu(self.bn1(self.conv1(x)))))
#         #print(x.shape)
#         x = self.drop(self.pool(F.relu(self.bn2(self.conv2(x)))))
#        # print(x.shape)
#         x = self.drop(self.pool(F.relu(self.bn3(self.conv3(x)))))
#       #  print(x.shape)
#         x = self.drop(self.pool(F.relu(self.bn4(self.conv4(x)))))
#        # print(x.shape)
#      #   x = self.drop(self.pool(F.relu(self.bn5(self.conv5(x)))))
#     #    print(x.shape) # TODO check x.shape for lines 50, 60
#         x = x.view(-1, 64*2*2) 
#         x = F.relu(self.fc(x))
#         x = self.fc1(x)
#         return x
#############################################################################
############################################################################
#############################################################################
# class MelSpecDataset(Dataset):
#   def __init__(self , csv_file , data_dir):
    
#     self.df = pd.read_csv(csv_file, header=None)
#     self.data_dir = data_dir
    
#   def __getitem__(self, idx):
#     d = self.df.iloc[idx]
#     ch1 = cv2.imread(os.path.join(d[0], d[0].split('/')[-1]+'_ch1.png'))
#     ch1 = cv2.cvtColor(ch1, cv2.COLOR_BGR2GRAY)
#     ch2 = cv2.imread(os.path.join(d[0], d[0].split('/')[-1]+'_ch2.png'))
#     ch2 = cv2.cvtColor(ch2, cv2.COLOR_BGR2GRAY )
#     ch1 = cv2.normalize(ch1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     ch2 = cv2.normalize(ch2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     arr = np.array([[ch1, ch2]], dtype=np.float32)
#     # trans = transforms.ToTensor()
#     # arr = trans(ch1)
#     arr = torch.from_numpy(arr)
#     label = torch.tensor(d[1])
#     return arr, label
  
#   def __len__(self):
#     return len(self.df)

class MelSpecDataset(Dataset):
  def __init__(self , csv_file , data_dir):
    
    self.df = pd.read_csv(csv_file, header=None)
    self.data_dir = data_dir
    
  def __getitem__(self, idx):
    d = self.df.iloc[idx]
    ch = cv2.imread(d[0])
    ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
    ch = cv2.normalize(ch, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cropped_imgs = [ch[:,i*24:i*24+24] for i in range(8)]
    arr = np.array([cropped_imgs], dtype=np.float32)
    # trans = transforms.ToTensor()
    # arr = trans(ch1)
    arr = torch.from_numpy(arr)
    arr = arr.unsqueeze(dim=2)
    label = torch.tensor(d[1])
    return arr, label
  
  def __len__(self):
    return len(self.df)
##############################################################################
train = MelSpecDataset(train_csv, img_dir)
val = MelSpecDataset(val_csv, img_dir)
test = MelSpecDataset(test_csv, img_dir)


batch_size = 32

# test_no = int(len(data)*0.15)
# val_no = int(len(data)*0.15)

# train, val, test = random_split(data, (len(data)-val_no-test_no, val_no, test_no), generator = torch.Generator().manual_seed(42))
print('Train, val, test samples: {}, {}, {}'.format(len(train), len(val), len(test)))

dataloader = {'train':DataLoader(train,
                                  shuffle = True ,
                                 batch_size = batch_size, num_workers=4),
              'val': DataLoader(val , 
                                shuffle = True ,
                                batch_size = batch_size, num_workers=4)}


lr = 0.001
nlayers = 1
hidden_size = 32
n_classes = 11
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

print("Extracting labels...\n")
labels = np.array([elm[1].item() for elm in train])
class_weight = compute_class_weight(class_weight='balanced', classes=np.array(list(range(n_classes))), y=labels)
class_weight = torch.tensor(class_weight, dtype=torch.float32)

cnn_encoder = irmasEncoder().to(device)
cnn_encoder.apply(initialize_weights_enc)
lstm_decoder = irmasDecoder(32*16*1, hidden_size, nlayers, n_classes).to(device)
lstm_decoder.apply(initialize_weights_dec)
# cnn_encoder = nn.DataParallel(cnn_encoder)
# lstm_decoder = nn.DataParallel(lstm_decoder)
combo_params = list(cnn_encoder.parameters()) + list(lstm_decoder.parameters())

# model = irmasCNN().to(device)
# model_summary = str(summary(model, input_size=(batch_size, 2, 124, 166), col_names=["output_size", "num_params", "kernel_size"], device=device, verbose=0))
# model = irmasLSTM(input_size, hidden_size, nlayers, n_classes).to(device)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss(weight=class_weight)
# optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.000001)
optimizer = torch.optim.Adam(combo_params, lr = lr)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=5,verbose=True)

#X_train= []
#y_train = []
#for i in range(len(data)):
#    X_train.append(data[i][0])
#    y_train.append(data[i][1])


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
        self.enc = None
        self.dec = None
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
                if savemodel:
                    save_model(self.epoch, self.enc, self.optimizer, r'../models/Checkpoints/CnnCheckpoint_{}_{}.pt'.format(self.epoch+1, time.time()), self.batch)
                    save_model(self.epoch, self.dec, self.optimizer, r'../models/Checkpoints/LSTMCheckpoint_{}_{}.pt'.format(self.epoch+1, time.time()), self.batch)
                self.early_stop = True
    
    def keepmodel(self, epoch, model, optimizer, batch, ytrues, ypreds, classes_num):
        enc, dec = model

        self.epoch = epoch
        self.enc = enc
        self.dec = dec
        self.optimizer = optimizer
        self.batch = batch
        self.ytrues = ytrues
        self.ypreds = ypreds
        self.classes_num = classes_num

def model_train(model , data_loader , criterion , optimizer, scheduler, early_stopping, num_epochs = 30):
    enc, dec = model

    for epoch in trange(num_epochs, desc = 'Epochs'):
        result = []
        ytrues = [[],[]]
        ypreds = [[],[]]
        for phase in ['train', 'val']:
            if phase == "train":
                enc.train()
                dec.train()
            else:  
                enc.eval()
                dec.eval()
       
            running_loss = 0.0
            running_corrects = 0.0  
            for data , target in data_loader[phase]:
                data , target = data.to(device)  , target.to(device)
                data = torch.squeeze(data, 1).to(device)
                target = target.type(torch.LongTensor).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    output = dec(enc(data))
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
                        early_stopping(epoch_loss, epoch, [enc, dec], optimizer, batch_size, ytrues, ypreds, n_classes)
            
            if phase == 'val':
                scheduler.step(epoch_loss)

        print(result)

        if writestats:
            if epoch + 1 == num_epochs and early_stopping.early_stop == False:
                stats(ytrues[0], ypreds[0], ytrues[1], ypreds[1], n_classes, epoch)

        if savemodel:
            if epoch + 1 == num_epochs and early_stopping.early_stop == False:
                try:
                    save_model(epoch, model, optimizer, r'../models/Checkpoints/CnnCheckpoint_{}_{}.pt'.format(epoch+1, time.time()), batch_size)
                    save_model(epoch, model, optimizer, r'../models/Checkpoints/LSTMCheckpoint_{}_{}.pt'.format(epoch+1, time.time()), batch_size)
                except:
                    print('Error: Model could not be saved!')

    return model



early_stopping = EarlyStopping(patience=7)

# cnn = model_train(model, dataloader, criterion, optimizer, early_stopping, num_epochs = 100)
cnn = model_train([cnn_encoder, lstm_decoder], dataloader, criterion, optimizer, lr_scheduler, early_stopping, num_epochs = 100)

# with open('../stats/model_summary.txt', 'w') as f:
#     f.write(model_summary)

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

'''
###########################  confusion  matrix and test F1
test_csv = r'C:\Datasets\Irmas\IRMAS-TrainingData\train'
test_dir = r'C:\Datasets\Irmas\tests'

test_data = MultitestDataset(test_csv, test_dir)

testloader = DataLoader(test_data, shuffle = True, batch_size = 1)
dataiter = iter(testloader)
images, labels = dataiter.next()

cnn.eval()

y_true, y_pred = [], []
with torch.no_grad():
    aggr = []
    for data in testloader:
        images, label = data
        y_true.append(label)
        for image in images:
            output = cnn(images)   
            aggr.append(output)
        y_pr = np.array(aggr)
        y_pr = np.mean(y_pr, axis = 1)
        y_pr = [1 if i > 0.3 else 0 for i in y_pr]
        y_pr = y_pr/np.max(y_pr)
        y_pred.append(y_pr)
    
conf = confusion_matrix(y_true, y_pred)
plt.matshow(conf)       
test_f1 = f1_score(y_true, y_pred, average = 'macro' , labels=np.unique(y_pred))
print('F1 score on test data: ', test_f1)       
'''

################ROC curves

'''
Good due to high FP !!!

'''
'''
y_tr, y_pr = [], []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        label = [0]*11
        label[int(labels)] = 1
        y_tr.append(label)
        y_pr.append(np.array(F.softmax(outputs).tolist()[0]))

y_test = np.array(y_tr)
y_pred = np.array(y_pr)
    
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(11):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])   
labels = 'cel cla flu gac gel org pia sax tru vio voi'.split()

plt.figure()
colors = {}
for i in range(11):
    r = random.random()
    g = random.random()
    b = random.random()
    color = (r, g, b)
    colors[i] = color
 
for i in range(11):
    plt.plot(fpr[i], tpr[i], color = colors[i],
         lw = 2, 
         label = f'{labels[i]}, (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC OneVsAll')
plt.legend(loc = 'lower right')
plt.show()
'''






