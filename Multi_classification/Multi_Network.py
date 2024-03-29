# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-20 17:38

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    用于"纲"与"种"多标签分类的网络
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        # 二分类：是哺乳纲还是鸟纲
        self.fc1 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)
        # 三分类：是兔子、老鼠还是鸡
        self.fc2 = nn.Linear(150, 3)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_classes = self.fc1(x)
        x_classes = self.softmax1(x_classes)

        x_species = self.fc2(x)
        x_species = self.softmax2(x_species)
        return x_classes, x_species
