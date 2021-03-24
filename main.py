import argparse
import models
import glob
from PIL import Image
import torch
from torchvision import transforms
from DataUtils import load_data

parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
parser.add_argument('--dataset', type=str, default='tuberlin', metavar='N', help='tuberlin/miniquickdraw')
args = parser.parse_args()

# creo modello a 4 filtri convoluzionali e un fc finale
enc_nn = models.create_models(args=args)
print(enc_nn)

# python3 main.py --dataset tuberlin
if (args.dataset == 'tuberlin'):
    images = glob.glob("./TUBerlin/train/airplane/1.png")
    for image in images:
        img = Image.open(image)
        img = transforms.ToTensor()(img).unsqueeze(0)
        enc_nn.eval()
        out = enc_nn(img)
        print(out)


# python3 main.py --dataset miniquickdraw
elif (args.dataset == 'miniquickdraw'):
    dataset = load_data.QD_Dataset('train')
    imm = dataset.__getitem__(0)[0]
    imm = torch.reshape(imm, (1, 1, 28, 28))
    enc_nn.eval()
    out = enc_nn(imm)
    print(out)
