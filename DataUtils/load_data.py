import os
import numpy as np
import torch
import torch.utils.data as data


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


class QD_Dataset(data.Dataset):
    def __init__(self, mtype, root='./DataUtils/MiniQuickDraw'):
        """
        args:
        - mytpe: str, specify the type of the dataset, i.e, 'train' or 'test'
        - root: str, specify the root of the dataset directory
        """

        self.data1, self.data2, self.target, self.num_classes = load_dataset(root, mtype)
        self.data1 = torch.from_numpy(self.data1)
        self.data2 = torch.from_numpy(self.data2)
        self.target = torch.from_numpy(self.target)
        print("Dataset " + mtype + " loading done.")
        print("*" * 50 + "\n")

    def __getitem__(self, index):
        imm1 = self.data1[index]
        imm1 = torch.reshape(imm1, (1, 28, 28))
        target1 = self.target[index]
        if index % 2 == 0:
            imm2 = self.data2[(index + 1) % (len(self.data2))]
            target2 = self.target[(index + 1) % (len(self.target))]
        else:
            imm2 = self.data2[(index + 300) % (len(self.data2))]
            target2 = self.target[(index + 300) % (len(self.target))]
        imm2 = torch.reshape(imm2, (1, 28, 28))
        if target1 == target2:
            label = 1.0
        else:
            label = 0.0
        return imm1, imm2, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return len(self.data1)

    def get_number_classes(self):
        return self.num_classes
