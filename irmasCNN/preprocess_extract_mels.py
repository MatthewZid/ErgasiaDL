import os
import sys
import librosa as lb
import librosa.display as ld
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import time

typename = str(sys.argv[1])

HOP = 128
WINDOW = 1024
DPI = 26
writetocsv = True
augment = False

datapath = '../csv/'+typename+'_dataset.csv'
melpath = '../meldata_stereo_masked/'+typename
melcsv = '../csv/'+typename+'_mel_stereo_dataset.csv'
dataset = pd.read_csv(datapath, header=None)
random.seed(42)

if writetocsv: open(melcsv, 'w').close()

def saveplot(sdb, sr, dirpath, melname, ch):
    fig = plt.figure(1)
    ld.specshow(sdb, sr=sr, hop_length=HOP)
    if writetocsv: fig.savefig(os.path.join(dirpath, '{}_{}.png'.format(melname, ch)), transparent=True, dpi=DPI)
    fig.clf()

def split_overlap(y, sr, overlap, time=1.5):
    start = 0
    end = time*sr
    split_y = []
    while(end <= y.shape[1]):
        split_y.append(y[:, int(start):int(end)])
        start = end - overlap
        end = start + time*sr
    
    return split_y

def freq_mask(mel, F=27, num_masks=1):
    mel_masked = mel.copy()
    num_channels = mel_masked.shape[0]

    for _ in range(num_masks):
        freq = random.randrange(0, F)
        zero = random.randrange(0, num_channels - freq)
        if (zero == zero + freq): return mel_masked
        mask_end = random.randrange(zero, zero + freq) 
        mel_masked[int(zero):int(mask_end), :] = mel_masked.mean()
    
    return mel_masked

def time_mask(mel, time=40, num_masks=1):
    mel_masked = mel.copy()
    length = mel_masked.shape[1]
    for _ in range(0, num_masks):
        t = random.randrange(0, time)
        zero = random.randrange(0, length - t)
        if (zero == zero + t): return mel_masked
        mask_end = random.randrange(zero, zero + t)
        mel_masked[:, int(zero):int(mask_end)] = mel_masked.mean()

    return mel_masked

print('Creating mels...')
start_time = time.time()

for _, row in dataset.iterrows():
    y, sr = lb.load(row[0], mono=False)
    # y[[0,1]] = y[[1,0]] # swap channels
    # np.random.shuffle(y) # randomly shuffle channels

    melname = row[0].split('/')[-1].split('.')[0]
    dirpath = os.path.join(melpath, melname)
    if writetocsv:
        try: os.makedirs(dirpath)
        except OSError as error:
            print("Error creating dir: {}".format(error))
            continue

    mel1 = lb.feature.melspectrogram(y[0], sr=sr, n_fft=WINDOW, hop_length=HOP)
    S_dB1 = lb.power_to_db(mel1, ref=np.max)
    saveplot(S_dB1, sr, dirpath, melname, 'ch1')

    mel2 = lb.feature.melspectrogram(y[1], sr=sr, n_fft=WINDOW, hop_length=HOP)
    S_dB2 = lb.power_to_db(mel2, ref=np.max)
    saveplot(S_dB2, sr, dirpath, melname, 'ch2')

    if augment:
        # masked mel
        if writetocsv:
            try: os.makedirs(dirpath+'_masked')
            except OSError as error:
                print("Error creating dir: {}".format(error))
                continue

        mel1_masked = freq_mask(S_dB1, num_masks=2)
        mel1_masked = time_mask(mel1_masked, num_masks=2)
        saveplot(mel1_masked, sr, dirpath+'_masked', melname+'_masked', 'ch1')

        mel2_masked = freq_mask(S_dB2, num_masks=2)
        mel2_masked = time_mask(mel2_masked, num_masks=2)
        saveplot(mel2_masked, sr, dirpath+'_masked', melname+'_masked', 'ch2')

    if writetocsv:
        with open(melcsv, 'a', encoding='utf-8') as csvf:
            csvwriter = csv.writer(csvf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([dirpath, row[1]])
            if augment: csvwriter.writerow([dirpath+'_masked', row[1]])

print('Finished!')
end_time = time.time()
print('Time elapsed: {:.2f} mins'.format((end_time - start_time)/60.0)) 
