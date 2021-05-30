import torch
import config
from torch.utils.data import DataLoader
from models import EmbeddingTUBerlin, EmbeddingMiniQuickDraw
from dataset_n_way import NWayOneShotEvalSetTUBerlin, NWayOneShotEvalSetMiniQuickDraw
import make_table

results=[]
nWays = config.nWays
datasets = ['tuberlin', 'miniquickdraw']
for dataset in datasets:
    print("Evaluation model with " + dataset+":")
    for nWay in nWays:
        if 'tuberlin' == dataset:
            model = EmbeddingTUBerlin()
            model.load_state_dict(torch.load("./tuberlin_model.pt"))
            model.eval()
            test_dataset = NWayOneShotEvalSetTUBerlin(nWay, config.TUBerlin + "val.csv", config.TUBerlin)
        elif 'miniquickdraw' == dataset:
            model = EmbeddingMiniQuickDraw()
            model.load_state_dict(torch.load("./miniquickdraw_model.pt"))
            model.eval()
            test_dataset = NWayOneShotEvalSetMiniQuickDraw('test', nWay)
        else:
            raise NameError('Dataset ' + dataset + ' not knows')
        test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=True)
        with torch.no_grad():
            model.eval()
            correct = 0
            count = 0
            for mainImg, imgSets, label in test_dataloader:
                predVal = 0
                pred = -1
                for i, testImg in enumerate(imgSets):
                    output = model(mainImg, testImg)
                    if output > predVal:
                        pred = i
                        predVal = output
                if pred == label:
                    correct += 1
                count += 1
                if count % 12 == 0:
                    acc=round(correct/count, 2)
                    print('Accuracy on {}-Way: {}'.format(nWay, acc))
                    results.append(acc)
                    break

header = ['Datasets:', '2-Way', '5-Way', '10-Way']
names = ['TUBerlin', 'MiniQuickDraw']
Way2 = [str(results[0]), str(results[3])]
Way5 = [str(results[1]), str(results[4])]
Way10 = [str(results[2]), str(results[5])]
make_table.makeTable(header, [names, Way2, Way5, Way10])
