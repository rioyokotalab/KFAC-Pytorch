import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

import sys
import numpy as np

from collections import OrderedDict

# set seed
torch.manual_seed(1234)

def load_testCNN(channels_one=32, channels_two=64, dropout=0.2):
    # test CNN
    # CNNモデルの構築。今回はtest用に3層
    # (Conv Relu | Conv Relu (dropout)| Affine relu | Affine (Softmax))
    # nn.Moduleを継承している。
    class testCNNNet(nn.Module):
        def __init__(self):
            super(testCNNNet, self).__init__()
            # Conv2d(in_channels, out_channels(), kernel_size(フィルター), stride=1, padding=0,
            # dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.conv1 = nn.Conv2d(3, channels_one, 3)  # 32x32x3 -> 30x30x32
            self.conv2 = nn.Conv2d(channels_one, channels_two, 3)  # 30x30x32 -> 28*28*64
            self.dropout1 = nn.Dropout2d(dropout)
            self.fc1 = nn.Linear(28 * 28 * channels_two, 1024)
            self.fc2 = nn.Linear(1024, 10)

        def forward(self, x):  # predictに相当(順伝搬)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.dropout1(x)
            x = x.view(-1, 28 * 28 * channels_two)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return testCNNNet()

def load_VGG16_bn():
    # VGG16 with batch normalize
    return models.vgg16_bn()

def load_VGG16():
    # VGG16 with batch normalize
    return models.vgg16()

def test_model():
    class CNNNet(nn.Module):
        def __init__(self):  # , num_classes):
            super(CNNNet, self).__init__()
            num_classes = 10

            self.block1_output = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.block2_output = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.block3_output = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.block4_output = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.block5_output = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # 512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 32),  # 4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(32, num_classes),  # 4096
            )

        def forward(self, x):
            x = self.block1_output(x)
            x = self.block2_output(x)
            x = self.block3_output(x)
            x = self.block4_output(x)
            x = self.block5_output(x)
            # print(x.size())
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return CNNNet()

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out



def WRN(depth, widen_factor, dropout_rate, num_classes):
    class Wide_ResNet(nn.Module):
        def __init__(self, depth, widen_factor, dropout_rate, num_classes):
            super(Wide_ResNet, self).__init__()
            self.in_planes = 16

            assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
            n = (depth-4)/6
            k = widen_factor

            # print('| Wide-Resnet %dx%d' %(depth, k))
            nStages = [16, 16*k, 32*k, 64*k]

            self.conv1 = conv3x3(3,nStages[0])
            self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
            self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
            self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
            self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
            self.linear = nn.Linear(nStages[3], num_classes)

        def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
            strides = [stride] + [1]*(int(num_blocks)-1)
            layers = []

            for stride in strides:
                layers.append(block(self.in_planes, planes, dropout_rate, stride))
                self.in_planes = planes

            #return nn.Sequential(*layers)
            return nn.Sequential(OrderedDict([("wide_basic_%d"%i, layer) for (i,layer) in enumerate(layers)]))

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            return out

    return Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)
