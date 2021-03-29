import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTUBerlin(nn.Module):
    def __init__(self, args, emb_size):
        ''' In this network the input image is supposed to be 256x256 '''

        super(EmbeddingTUBerlin, self).__init__()
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
        enc_nn = EmbeddingTUBerlin(args, 64)
    elif 'miniquickdraw' == args.dataset:
        enc_nn = EmbeddingMiniQuickDraw(args, 64)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn
