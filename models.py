import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTUBerlin(nn.Module):
    ''' In this network the input image is supposed to be 84x84 '''

    def __init__(self, args, emb_size):
        super(EmbeddingTUBerlin, self).__init__()
        self.emb_size = emb_size
        self.ndf = 64
        self.args = args

        # Input 84x84x1
        self.conv1 = nn.Conv2d(1, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf * 1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf * 4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf * 4 * 5 * 5, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        x = x.view(-1, self.ndf * 4 * 5 * 5)
        output = self.bn_fc(self.fc1(x))

        return [e1, e2, e3, e4, None, output]




class EmbeddingMiniQuickDraw(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, args, emb_size):
        super(EmbeddingMiniQuickDraw, self).__init__()
        self.emb_size = emb_size
        self.nef = 64
        self.args = args

        # input is 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, self.nef, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nef)
        # state size. (nef) x 14 x 14
        self.conv2 = nn.Conv2d(self.nef, self.nef, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nef)

        # state size. (1.5*ndf) x 7 x 7
        self.conv3 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 5 x 5
        self.conv4 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 3 x 3
        self.fc_last = nn.Linear(3 * 3 * self.nef, self.emb_size, bias=False)
        self.bn_last = nn.BatchNorm1d(self.emb_size)

    def forward(self, inputs):
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs)), 2)
        x = F.leaky_relu(e1, 0.1, inplace=True)

        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.1, inplace=True)

        e3 = self.bn3(self.conv3(x))
        x = F.leaky_relu(e3, 0.1, inplace=True)
        e4 = self.bn4(self.conv4(x))
        x = F.leaky_relu(e4, 0.1, inplace=True)
        x = x.view(-1, 3 * 3 * self.nef)

        output = F.leaky_relu(self.bn_last(self.fc_last(x)))

        return [e1, e2, e3, output]


def create_models(args):
    print(args.dataset)
    if 'tuberlin' == args.dataset:
        enc_nn = EmbeddingTUBerlin(args, 128)
    elif 'miniquickdraw' == args.dataset:
        enc_nn = EmbeddingMiniQuickDraw(args, 64)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn
