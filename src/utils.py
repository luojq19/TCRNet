# -*- codeing = utf-8 -*-
# @Time:  9:18 下午
# @Author: Jiaqi Luo
# @File: utils.py
# @Software: PyCharm

import torch
import numpy as np

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Blosum matrix citation: Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of
# TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021).
# https://doi.org/10.1038/s42003-021-02610-3
blosum50_20aa = {
    'A': torch.tensor((5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)),
    'R': torch.tensor((-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3)),
    'N': torch.tensor((-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3)),
    'D': torch.tensor((-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4)),
    'C': torch.tensor((-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)),
    'Q': torch.tensor((-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3)),
    'E': torch.tensor((-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3)),
    'G': torch.tensor((0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)),
    'H': torch.tensor((-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4)),
    'I': torch.tensor((-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4)),
    'L': torch.tensor((-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1)),
    'K': torch.tensor((-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3)),
    'M': torch.tensor((-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1)),
    'F': torch.tensor((-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1)),
    'P': torch.tensor((-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)),
    'S': torch.tensor((1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2)),
    'T': torch.tensor((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)),
    'W': torch.tensor((-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)),
    'Y': torch.tensor((-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)),
    'V': torch.tensor((0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5))
}

# Encode one amino acid sequence to one-hot form with expected length (with zero padding)
def SeqToOneHot(seq, expected_len=40):
    encoding = torch.zeros([20, expected_len], dtype=torch.float32)
    seq = seq.strip()
    for i in range(len(seq)):
        if seq[i] not in amino_acids:
            continue
        encoding[amino_acids.index(seq[i]), i] = 1.0

    return [encoding.tolist()]

# Encode one amino acid sequence to Blosum50 form with expected length (with zero padding)
def SeqToBlosum50(seq, expected_len=40):
    encoding = torch.zeros([20, expected_len], dtype=torch.float32)
    seq = seq.strip()
    for i in range(len(seq)):
        if seq[i] not in amino_acids:
            continue
        encoding[:, i] = blosum50_20aa[seq[i]].T

    return [encoding.tolist()]

# : one-hot encoding
# Input: the txt data file and sample type (pos or neg)
# Output:  the corresponding one hot encodings of all the sequences and
# the corresponding labels (1 for positive samples and 0 for negative samples
def EncodeOneHot(data_path, sample_type):
    assert ((sample_type == "positive" or sample_type == "negative") and (sample_type in data_path))
    X, y = [], []
    with open(data_path, "r") as f:
        sequences = f.readlines()
        sequences = list(set(sequences))
        for seq in sequences:
            X.append(SeqToOneHot(seq))
        if sample_type == "positive":
            y = torch.ones((len(sequences), 1), dtype=torch.float32)
        else:
            y = torch.zeros((len(sequences), 1), dtype=torch.float32)

    return torch.tensor(X), y


# : Blosum50 encoding
# Input: the txt data file and sample type (pos or neg)
# Output:  the corresponding Blosum50 encodings of all the sequences and
# the corresponding labels (1 for positive samples and 0 for negative samples
def EncodeBlosum50(data_path, sample_type):
    assert ((sample_type == "positive" or sample_type == "negative") and (sample_type in data_path))
    X, y = [], []
    with open(data_path, "r") as f:
        sequences = f.readlines()
        sequences = list(set(sequences))
        for seq in sequences:
            X.append(SeqToBlosum50(seq))
        if sample_type == "positive":
            y = torch.ones((len(sequences), 1), dtype=torch.float32)
        else:
            y = torch.zeros((len(sequences), 1), dtype=torch.float32)

    return torch.tensor(X), y

def Shuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a.cpu().detach().numpy())
    np.random.set_state(state)
    np.random.shuffle(b.cpu().detach().numpy())

    return torch.tensor(a), torch.tensor(b)

# : implement k-fold for the given data
def GetKfoldData(k, i, X, y):
    assert k > 1
    X, y = Shuffle(X, y)
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid