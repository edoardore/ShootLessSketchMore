import os
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd

import config


class NWayOneShotEvalSetTUBerlin():
    def __init__(self, nWay, training_csv=None, training_dir=None):
        # used to prepare the labels and images path
        self.nWay = nWay
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir

    def openImage(self, path):
        img0 = Image.open(path)
        basewidth = 105
        wpercent = (basewidth / float(img0.size[0]))
        hsize = int((float(img0.size[1]) * float(wpercent)))
        img0 = img0.resize((basewidth, hsize), Image.ANTIALIAS)
        img0 = transforms.ToTensor()(img0)
        return img0

    def __getitem__(self, index):
        while (int(self.train_df.iat[index, 2]) != 1):
            index = (index + np.random.randint(len(self.train_df))) % len(self.train_df)

        main_path = os.path.join(self.train_dir, self.train_df.iat[index, 0]).rstrip("\n")
        image_path = os.path.join(self.train_dir, self.train_df.iat[index, 1]).rstrip("\n")

        # getting the image path
        paths = []
        classes = []
        classes.append(image_path.split('/')[2])
        while len(paths) < self.nWay:
            image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0]).rstrip("\n")
            index = (index + np.random.randint(len(self.train_df))) % len(self.train_df)
            clas = image1_path.split('/')
            if clas[2] not in classes:
                classes.append(clas[2])
                paths.append(image1_path)

        main_image = self.openImage(main_path)
        testImages = []
        label = np.random.randint(self.nWay)

        for i in range(0, len(paths)):
            if i == label:
                testImages.append(self.openImage(image_path))
            else:
                testImages.append(self.openImage(paths[i]))

        return main_image, testImages, torch.from_numpy(np.array([label], dtype=int))

    def __len__(self):
        return len(self.train_df)


def load_dataset(root, mtype):
    num_classes = 0
    with open("./DataUtils/class_names.txt", "r") as f:
        for line in f:
            num_classes = num_classes + 1

    # load data from cache
    if os.path.exists(os.path.join(root, mtype + '.npz')):
        print("*" * 50)
        print("Loading " + mtype + " dataset...")
        print("*" * 50)
        print("Classes number of " + mtype + " dataset: " + str(num_classes))
        print("*" * 50)
        data_cache = np.load(os.path.join(root, mtype + '.npz'))
        return data_cache["data1"].astype('float32'), data_cache["data2"].astype('float32'), data_cache[
            "target"].astype(
            'int64'), num_classes
    else:
        raise FileNotFoundError("%s doesn't exist!" %
                                os.path.join(root, mtype + '.npz'))


class NWayOneShotEvalSetMiniQuickDraw():
    def __init__(self, mtype, root='./DataUtils/MiniQuickDraw'):
        self.data1, self.data2, self.target, self.num_classes = load_dataset(root, mtype)
        self.data1 = torch.from_numpy(self.data1)
        self.data2 = torch.from_numpy(self.data2)
        self.target = torch.from_numpy(self.target)
        self.nWay = config.nWay
        print("Dataset " + mtype + " loading done.")
        print("*" * 50 + "\n")

    def __getitem__(self, index):
        imm1 = self.data1[index]
        imm1 = torch.reshape(imm1, (1, 28, 28))
        imm2 = self.data2[(index + 1) % (len(self.data2))]
        imm2 = torch.reshape(imm2, (1, 28, 28))

        testImm = []
        while len(testImm) < self.nWay:
            imm = self.data2[(index + 200) % (len(self.data2))]
            imm = torch.reshape(imm, (1, 28, 28))
            testImm.append(imm)

        label = np.random.randint(self.nWay)
        swapImm = testImm[label]
        testImm[label] = imm2
        testImm.append(swapImm)

        return imm1, testImm, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return len(self.data1)

    def get_number_classes(self):
        return self.num_classes
