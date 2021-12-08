# -*- codeing = utf-8 -*-
# @Time:  9:19 下午
# @Author: Jiaqi Luo
# @File: train.py
# @Software: PyCharm

import torch
import numpy as np
from torch.utils import data
from sklearn.metrics import roc_auc_score
from src.utils import *

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# : training CNN model
def train(model, train_features, train_labels, optimizer, loss, num_epochs, lr, batch_size, device):
    train_iter = load_array((train_features, train_labels), batch_size)
    model.to(device)
    for epoch in range(num_epochs):
        # training
        sum_loss = 0.0
        train_correct = 0
        for datum in train_iter:
            inputs, labels = datum
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            sum_loss += l.item()
            train_correct += float(((outputs - labels).abs() < 0.5).sum().item())

        print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss * batch_size / len(train_features)))
        print('        correct:%.03f%%' % (100 * train_correct / len(train_features)))

    return model

def kFold(model, k, train_features, train_labels, optimizer, loss, num_epochs, lr, batch_size, device):
    auc_scores = []
    for i in range(k):
        X_train, y_train, X_valid, y_valid = GetKfoldData(k, i, train_features, train_labels)
        net = model
        print("Fold %d: " % (i + 1))
        net = train(net, X_train, y_train, optimizer, loss, num_epochs, lr, batch_size, device)
        X_valid = X_valid.to(device)
        y_pred = net(X_valid).cpu()
        auc = roc_auc_score(y_valid.detach().numpy(), y_pred.detach().numpy())
        print("AUC score:", auc)
        auc_scores.append(auc)
    for i in range(k):
        print("Fold %d AUC score = %.4f" % (i + 1, auc_scores[i]))
    print("Average AUC score = %.4f" % (torch.tensor(auc_scores).mean().item()))

    return auc_scores
