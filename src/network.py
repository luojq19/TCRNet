# -*- codeing = utf-8 -*-
# @Time:  9:19 下午
# @Author: Jiaqi Luo
# @File: network.py
# @Software: PyCharm

import torch
from torch import nn
from torch.nn import functional as F

# : construct the convolution network
class TCRNet(nn.Module):
    def __init__(self):
        super(TCRNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=1, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=5, stride=1, padding="same")
        self.conv7 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=7, stride=1, padding="same")
        self.conv9 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=9, stride=1, padding="same")

        self.conv2_1 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=1, stride=1, padding="same")
        self.conv2_3 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=1, stride=1, padding="same")
        self.conv2_5 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=1, stride=1, padding="same")
        self.conv2_7 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=1, stride=1, padding="same")
        self.conv2_9 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=1, stride=1, padding="same")

        self.max_pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool3 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool5 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool7 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool9 = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(nn.Linear(500, 10),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(10, 1))

    def forward(self, x):
        c1 = self.max_pool1(F.relu(self.conv2_1(F.relu(self.conv1(x)))))
        c3 = self.max_pool3(F.relu(self.conv2_3(F.relu(self.conv3(x)))))
        c5 = self.max_pool5(F.relu(self.conv2_5(F.relu(self.conv5(x)))))
        c7 = self.max_pool7(F.relu(self.conv2_7(F.relu(self.conv7(x)))))
        c9 = self.max_pool9(F.relu(self.conv2_9(F.relu(self.conv9(x)))))
        concat = torch.cat((c1, c3, c5, c7, c9), dim=1)

        return self.fc(concat.view((-1, 500)))