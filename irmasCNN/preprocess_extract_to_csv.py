import os
import csv

datapath = r'../irmas/IRMAS-TrainingData'
csvpath = r'../csv'
writetocsv = True

dirlist = [os.path.join(datapath, name) for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))]
if writetocsv: open(os.path.join(csvpath, '../irmas_train.csv'), 'w').close()

class_num = 0
for dir in dirlist:
    filenames = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
    for fn in filenames:
        dirpath = os.path.join(dir, fn)

        if writetocsv:
            with open(os.path.join(csvpath, 'irmas_train.csv'), 'a', encoding="utf-8") as csvf:
                csvwriter = csv.writer(csvf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([dirpath, class_num])

    filenames.clear()
    class_num += 1 
