import config
import csv
import random
import linecache
from os import path

num_classes = config.numClass

print("-------Splitting files in train.csv------")
with open(config.TUBerlin + 'train.csv', 'w', newline='\n', encoding='utf-8') as file:
    for i in range(0, 8000):
        idxs = random.sample(range(num_classes * 80), 1)
        idxs.append(idxs[0] + 80)
        lines = [linecache.getline(config.TUBerlin + "filelist.txt", i) for i in idxs]
        image1 = lines[0]
        image2 = lines[1]
        x = image1.split('/')
        y = image2.split('/')
        if x[0] == y[0]:
            label = 1
        else:
            label = 0
        writer = csv.writer(file)
        if path.exists("./TUBerlin/" + image1.replace('\n', '')) and path.exists(
                "./TUBerlin/" + image2.replace('\n', '')):
            writer.writerow([image1.replace('\n', ''), image2.replace('\n', ''), label])
    for i in range(0, 8000):
        idxs = random.sample(range(num_classes * 80), 1)
        idxs.append(idxs[0] + 1)
        lines = [linecache.getline(config.TUBerlin + "filelist.txt", i) for i in idxs]
        image1 = lines[0]
        image2 = lines[1]
        x = image1.split('/')
        y = image2.split('/')
        if x[0] == y[0]:
            label = 1
        else:
            label = 0
        writer = csv.writer(file)
        if path.exists("./TUBerlin/" + image1.replace('\n', '')) and path.exists(
                "./TUBerlin/" + image2.replace('\n', '')):
            writer.writerow([image1.replace('\n', ''), image2.replace('\n', ''), label])

print("-------Splitting files in val.csv------")
with open(config.TUBerlin + 'val.csv', 'w', newline='\n', encoding='utf-8') as file:
    for i in range(0, 100):
        idxs = random.sample(range(num_classes * 80), 1)
        idxs.append(idxs[0] + 80)
        lines = [linecache.getline(config.TUBerlin + "filelist.txt", i) for i in idxs]
        image1 = lines[0]
        image2 = lines[1]
        x = image1.split('/')
        y = image2.split('/')
        if x[0] == y[0]:
            label = 1
        else:
            label = 0
        writer = csv.writer(file)
        if path.exists("./TUBerlin/" + image1.replace('\n', '')) and path.exists(
                "./TUBerlin/" + image2.replace('\n', '')):
            writer.writerow([image1.replace('\n', ''), image2.replace('\n', ''), label])
    for i in range(0, 100):
        idxs = random.sample(range(num_classes * 80), 1)
        idxs.append(idxs[0] + 1)
        lines = [linecache.getline(config.TUBerlin + "filelist.txt", i) for i in idxs]
        image1 = lines[0]
        image2 = lines[1]
        x = image1.split('/')
        y = image2.split('/')
        if x[0] == y[0]:
            label = 1
        else:
            label = 0
        writer = csv.writer(file)
        if path.exists("./TUBerlin/" + image1.replace('\n', '')) and path.exists(
                "./TUBerlin/" + image2.replace('\n', '')):
            writer.writerow([image1.replace('\n', ''), image2.replace('\n', ''), label])
print("-------Finished!------")
