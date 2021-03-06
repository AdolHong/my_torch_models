# coding=utf-8
import torch
import torch.nn as nn

class LogisticClassifier(nn.Module):
    def __init__(self, num_feature):
        super(LogistcClassifier, self).__init__()
        self.dense = nn.Linear(num_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        z = self.dense(X)
        logit = self.sigmoid(z)
        return logit

    def loss(self, X, y):
        logit = self.forward(X)
        criterion = nn.BCELoss()
        loss = criterion(logit, y)
        return loss

class SoftmaxClassifier(nn.Module):
    def __init__(self, num_feature, num_class):
        super(SoftmaxClassifier, self).__init__()
        self.dense = nn.Linear(num_feature, num_class)

    def forward(self, X):
        z = self.dense(X)
        logits = torch.softmax(z, dim=1)
        return logits

    def loss(self, X, y):
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        return loss
