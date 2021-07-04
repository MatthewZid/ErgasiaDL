# -*- coding: utf-8 -*-
import librosa 
import librosa.display
import numpy as np
import os
import pandas as pd
from scipy.signal import lfilter

def delta(data, axis=-1, width=9, order=1, trim=True): # calculate the delta(order=1)
                                                       #  or ddelta(order=2) of spec
    half_length     = 1 + int(np.floor(width / 2))
    window          = np.arange(half_length, - half_length - 1, -1)
    
    padding         = [(0, 0)]  * data.ndim
    padding[axis]   = (half_length, half_length)
    data            = np.pad(data, padding, mode='edge')

    delta_x = data
    
    for i in range(order):
        delta_x = lfilter(window, 1, delta_x, axis=axis)
    
    if trim:
        idx = [Ellipsis] * delta_x.ndim
        idx[axis] = slice(half_length, -half_length)
        delta_x = delta_x[idx]
    
    return delta_x

data_dir = r'C:\Datasets\Irmas\IRMAS-TrainingData\train'
labels = 'cel cla flu gac gel org pia sax tru vio voi'.split()
segmented_data_dir = r'C:\Datasets\Irmas\segmented'
os.makedirs(segmented_data_dir)
df = pd.DataFrame(columns=['image', 'integer'])

def one_hot(idx):
    label = [0]*11
    label[idx] = 1
    return label

def avg(tempo, N = 3):
    cum = np.cumsum(tempo, 0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = tempo.shape[0] % N
    if remainder != 0:
        if remainder < tempo.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

total_files = 0
for label in labels:
    print('Processing {}.'.format(label))
    label_path = os.path.join(data_dir, label, label)
    audios = librosa.util.find_files(label_path)
    target = labels.index(label)
    for file in audios:
        audio_path = os.path.join(label_path, file)
        audio, sr = librosa.load(audio_path)
        for i in (0, 1.5):
            segment = audio[int(i*sr) : int(i*sr + 1.5*sr)]
            filename = 'spec_{}_{}'.format(target, total_files)
            
            chroma = librosa.feature.chroma_stft(segment, sr, n_chroma = 24)
            mfccs = librosa.feature.mfcc(segment, sr, n_mfcc = 24)
            tempo = librosa.feature.fourier_tempogram(segment, sr, win_length = 128, hop_length = 520)
            tempo = avg(tempo) 
            ftr = np.concatenate((chroma, mfccs, tempo), axis = 0)
          #  print(ftr.shape)
            melspec = librosa.feature.melspectrogram(segment,
                                                 sr, 
                                                 n_fft = 1024,
                                                 hop_length = 259)
            #d = delta(melspec, order = 1)
            #dd = delta(melspec, order = 2)
                
           # stack = np.stack((melspec, d, dd), axis = 0)
            #print(melspec.shape)
            np.save(os.path.join(segmented_data_dir, filename), ftr)       
            df.loc[total_files] = [filename, target]
            total_files += 1
        
df = df.sample(frac=1)
df.to_csv(os.path.join(data_dir, '128x128.csv'))

total_files = len(df)
test_dir1 = r'C:\Datasets\Irmas\Part1'
test_dir2 = r'C:\Datasets\Irmas\Part2'
test_dir3 = r'C:\Datasets\Irmas\Part3'
test_dirs = (test_dir1, test_dir2, test_dir3)
test_dest = r'C:\Datasets\Irmas\test'
df1 = pd.DataFrame(columns=['images_dir', 'multi'])
df2 = df.copy()
def multi(targets):
    multi = 11*[0]
    for tar in targets:
        multi[tar] = 1
    return multi

total_files1 = 0
for direc in test_dirs:
    print('next dir')
    for file in os.listdir(direc):
        if file.endswith('.wav'):
            total_files2 = 0
            prefix = file[0: -4]
            txt_name = os.path.join(direc, prefix + '.txt')
            txt = open(txt_name, 'r')
            lines = txt.readlines()
            targets = []
            for line in lines:
                for c in labels:
                    if c in line:
                        tar = labels.index(c)
                        targets.append(tar)
                        break
            if len(targets) > 1:
                continue
            audiopath = os.path.join(direc, file)
            audio, sr = librosa.load(audiopath)
            for i in np.arange(0, len(audio)/sr - 1, 1.5):
                segment = audio[int(i*sr) : int(i*sr + 1.5*sr)]
                
                if len(targets) == 1:
                    filename = 'spec_{}_{}'.format(targets[0], total_files)
                else:
                    filename = 'spec_{}'.format(total_files2)
                    if total_files2 == 0:
                        os.makedirs(os.path.join(test_dest, prefix))
                melspec = librosa.feature.melspectrogram(segment,
                                                 sr, 
                                                 n_fft = 1024,
                                                 hop_length = 173)
               # d = delta(melspec, order = 1)
               # dd = delta(melspec, order = 2)
                
               # stack = np.stack((melspec, d, dd), axis = 0)
 
                if len(targets) == 1:
                    dest = os.path.join(segmented_data_dir, filename)
                    np.save(dest, melspec)                

                else:
                    dest = os.path.join(test_dest, prefix)
                    np.save(os.path.join(dest, filename), melspec)     
                if len(targets) == 1:
                    df.loc[total_files] = [filename, targets[0]]
                    total_files += 1
                else:
                    total_files2 += 1
            if len(targets) > 1:   
                df1.loc[total_files1] = [dest, multi(targets)]
                total_files1 += 1







df1 = df1.sample(frac=1)
df1.to_csv(os.path.join(r'C:\Datasets\Irmas', 'test.csv'), 
          index = False)

df = df.sample(frac=1)
df.to_csv(os.path.join(r'C:\Datasets\Irmas', 'train.csv'), 
          index = False)




