import pandas as pd
import os
import csv

testpath = r'../irmas/IRMAS-TestingData-Part'
datapath = r'../irmas/IRMAS-TrainingData'

dirlist = [name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))]

open('../csv/test_dataset.csv', 'w').close()

for i in range(1,4):
    dr = os.path.join(testpath+str(i), 'Part'+str(i))
    filelist = [os.path.join(dr, name) for name in os.listdir(dr) if os.path.isfile(os.path.join(dr, name)) and name.split('.')[-1] == 'txt']
    
    for fn in filelist:
        content = None
        with open(fn, 'r') as f:
            content = f.readlines()
        
        if content != None:
            if len(content) != 1: continue
            content = content[0].strip()
            with open('../csv/test_dataset.csv', 'a') as csvf:
                csvwriter = csv.writer(csvf, delimiter=',')
                pos = dirlist.index(content)
                csvwriter.writerow([os.path.splitext(fn)[0]+'.wav', pos]) 
