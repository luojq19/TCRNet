# -*- codeing = utf-8 -*-
# @Time:  1:52 下午
# @Author: Jiaqi Luo
# @File: UTest.py
# @Software: PyCharm

import torch
from sklearn.metrics import roc_auc_score
from src.utils import *
import math

# a = torch.tensor([[1, 1, 1], [1, 1, 1]])
# b = torch.tensor([[2, 2, 2], [2, 2, 2]])
# print(a)
# print(b)
# print(torch.cat([a, b], 0))
# print(torch.cat([a, b], 1))

# m = 0
# with open("../data/GIL_negative.txt", "r") as f:
#     s = f.readlines()
#     print(s[0].strip())
#     print(len(s))
#     print(len(list(set(s))))
#     for ss in s:
#         m = max(m, len(ss))
#     print(m)
# with open("../data/GIL_positive.txt", "r") as f:
#     s = f.readlines()
#     print(s[0].strip())
#     print(len(s))
#     print(len(list(set(s))))
#     for ss in s:
#         m = max(m, len(ss))
#     print(m)
# with open("../data/NLV_negative.txt", "r") as f:
#     s = f.readlines()
#     print(s[0].strip())
#     print(len(s))
#     print(len(list(set(s))))
#     for ss in s:
#         m = max(m, len(ss))
#     print(m)
# with open("../data/NLV_positive.txt", "r") as f:
#     s = f.readlines()
#     print(s[0].strip())
#     print(len(s))
#     print(len(list(set(s))))
#     for ss in s:
#         m = max(m, len(ss))
#     print(m)
# print("max len:", m)

# print(amino_acids)
# print(amino_acids.index('V'))

# a = torch.tensor([1.5, 1, 1])
# b = torch.tensor([2, 2, 2])
# print((a * b).sum().item())
# a = "aaabbb"
# for i in a:
#     print(i)
# print(SeqToBlosum50('CASSYPGGGFYEQY').shape)
# X, y = EncodeOneHot("../data/NLV_positive.txt", sample_type="positive")
# print(len(X), len(y))
# print(X[0], y[0])

# pred = torch.tensor([0.85, 0.91, 0.77, 0.72, 0.61, 0.48, 0.33, 0.42])
# # truth = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
# # auc = roc_auc_score(truth, pred)
# # print(auc)
# print(torch.tensor([1., 2., 3.]).mean().item())
# a = torch.zeros(2)
# b = torch.ones(3)
# print(torch.cat([b, b]))
x = [1, 1]
print(torch.tensor(x).shape, torch.tensor([x]).shape)