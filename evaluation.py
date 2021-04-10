import argparse
import torch
import config
from torch.utils.data import DataLoader
from models import EmbeddingTUBerlin, EmbeddingMiniQuickDraw
from dataset_n_way import NWayOneShotEvalSetTUBerlin, NWayOneShotEvalSetMiniQuickDraw

parser = argparse.ArgumentParser(description='Few-Shot Learning with Siamese Network')
parser.add_argument('--dataset', type=str, default='miniquickdraw', metavar='N', help='tuberlin/miniquickdraw')
args = parser.parse_args()

nWay = config.nWay

print("Evaluation model with " + args.dataset)
if 'tuberlin' == args.dataset:
    model = EmbeddingTUBerlin()
    model.load_state_dict(torch.load("./tuberlin_model.pt"))
    model.eval()
    test_dataset = NWayOneShotEvalSetTUBerlin(nWay, config.TUBerlin + "val.csv", config.TUBerlin)
elif 'miniquickdraw' == args.dataset:
    model = EmbeddingMiniQuickDraw()
    model.load_state_dict(torch.load("./miniquickdraw_model.pt"))
    model.eval()
    test_dataset = NWayOneShotEvalSetMiniQuickDraw('test')
else:
    raise NameError('Dataset ' + args.dataset + ' not knows')
test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=True)
with torch.no_grad():
    model.eval()
    correct = 0
    count = 0
    for mainImg, imgSets, label in test_dataloader:
        predVal = 0
        pred = -1
        # determine which category an image belongs to
        for i, testImg in enumerate(imgSets):
            output = model(mainImg, testImg)
            if output > predVal:
                pred = i
                predVal = output
        if pred == label:
            correct += 1
        count += 1
        if count % 20 == 0:
            print("Current Count is: {}".format(count))
            print('Accuracy on N-Way: {}'.format(correct / count))
