import  os
from PIL import Image
import pandas as pd


train_df = pd.read_csv("./TUBerlin/train.csv")
train_df.columns = ["image1", "image2", "label"]

for index in range (0, len(train_df)):
    print(str(train_df.iat[index, 0]))
    image1_path = os.path.join("./TUBerlin", str(train_df.iat[index, 0]))
    print(image1_path)
    img0 = Image.open(image1_path)
    print(index)


train_df = pd.read_csv("./TUBerlin/val.csv")
train_df.columns = ["image1", "image2", "label"]

for index in range (0, len(train_df)):
    print(str(train_df.iat[index, 0]))
    image1_path = os.path.join("./TUBerlin", str(train_df.iat[index, 0]))
    print(image1_path)
    img0 = Image.open(image1_path)
    print(index)
