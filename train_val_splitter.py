import config
import csv
import random
import linecache

num_classes = config.numClass

print("-------Splitting files in train.csv------")
with open(config.TUBerlin + 'train.csv', 'w', newline='') as file:
    for i in range(0, 50):
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
        writer.writerow([image1, image2, label])
    for i in range(0, 50):
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
        writer.writerow([image1, image2, label])

print("-------Splitting files in val.csv------")
with open(config.TUBerlin + 'val.csv', 'w', newline='') as file:
    for i in range(0, 20):
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
        writer.writerow([image1, image2, label])
    for i in range(0, 20):
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
        writer.writerow([image1, image2, label])
print("-------Finished!------")
