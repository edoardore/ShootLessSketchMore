import  os
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd

class SiameseDataset():
    def __init__(self, training_csv=None, training_dir=None):

        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_dir, str(self.train_df.iat[index, 0]))
        image2_path = os.path.join(self.train_dir, str(self.train_df.iat[index, 1]))
        # Loading the image
        img0 = Image.open(image1_path)
        basewidth = 105
        wpercent = (basewidth / float(img0.size[0]))
        hsize = int((float(img0.size[1]) * float(wpercent)))
        img0 = img0.resize((basewidth, hsize), Image.ANTIALIAS)
        img0 = transforms.ToTensor()(img0)

        img1 = Image.open(image2_path)
        basewidth = 105
        wpercent = (basewidth / float(img1.size[0]))
        hsize = int((float(img1.size[1]) * float(wpercent)))
        img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
        img1 = transforms.ToTensor()(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.train_df.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.train_df)
