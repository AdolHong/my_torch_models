# coding=utf-8
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, num_feature):
        super(LinearRegression, self).__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, X):
        y_hat = self.dense(X)
        return y_hat

    def loss(self, X, y):
        y_hat = self.forward(X)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        return loss


class LogisticRegression(nn.Module):
    def __init__(self, num_feature):
        super(LogistcRegression, self).__init__()
        self.dense = nn.Linear(num_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        z = self.dense(X)
        logit = self.sigmoid(z)
        return logit

    def loss(self, X, y):
        logit = self.forward(X)
        criterion = nn.MSELoss()
        loss = criterion(logit, y)
        return loss

