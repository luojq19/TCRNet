# -*- codeing = utf-8 -*-
# @Time:  3:49 下午
# @Author: Jiaqi Luo
# @File: main.py
# @Software: PyCharm

import numpy as np
import torch
from torch import nn
from src.utils import *
from src.train import *
from src.network import *

data_root = "../data/"
data_files = ["GIL_negative.txt", "GIL_positive.txt", "NLV_negative.txt", "NLV_positive.txt"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load training data
GIL_pos_feature, GIL_pos_label = EncodeOneHot(data_root + data_files[1], sample_type="positive")
GIL_neg_feature, GIL_neg_label = EncodeOneHot(data_root + data_files[0], sample_type="negative")
NLV_pos_feature, NLV_pos_label = EncodeOneHot(data_root + data_files[3], sample_type="positive")
NLV_neg_feature, NLV_neg_label = EncodeOneHot(data_root + data_files[2], sample_type="negative")

GIL_features = torch.cat((GIL_pos_feature, GIL_neg_feature))
GIL_labels = torch.cat([GIL_pos_label, GIL_neg_label])

NLV_features = torch.cat((NLV_pos_feature, NLV_neg_feature))
NLV_labels = torch.cat([NLV_pos_label, NLV_neg_label])

# DEBUG
# print(GIL_features.shape, GIL_features[0].shape, GIL_labels.shape)
# print(NLV_features.shape, NLV_features[0].shape, NLV_labels.shape)
# input()

# Define model
model = TCRNet()

# Hyperparameters
num_epochs = 10
batch_size = 64
lr = 0.001
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 5-fold training and cross validation
k = 5

print("Begin training on", device)
print("5 fold training and validation for GIL: ")
kFold(model, k, GIL_features, GIL_labels, optimizer, loss, num_epochs, lr, batch_size, device)

print("5 fold training and validation for NLV: ")
kFold(model, k, NLV_features, NLV_labels, optimizer, loss, num_epochs, lr, batch_size, device)


