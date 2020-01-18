import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import sklearn.preprocessing as sk
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.autograd import Function


#######################################################
#             AITL Classes & Functions                #          
#######################################################

class FX(nn.Module):
    def __init__(self, dropout_rate, input_dim, h_dim, z_dim):
        super(FX, self).__init__()
        self.EnE = torch.nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate))
    def forward(self, x):
        output = self.EnE(x)
        return output

class MTL(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(MTL, self).__init__()
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Source = torch.nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(z_dim, 1))
        self.Target = torch.nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid())        
    def forward(self, S, T):
        if S is None:
            ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = None
            yhat_T = self.Target(ZT)
        elif T is None:
            ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
            yhat_S = self.Source(ZS)
            yhat_T = None
        else: 
            ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
            ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = self.Source(ZS)
            yhat_T = self.Target(ZT)
        return yhat_S, yhat_T   

class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse()(x)

class Discriminator(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(Discriminator, self).__init__()
        self.D1 = nn.Linear(h_dim, 1)
        self.Drop1 = nn.Dropout(p=dropout_rate)
        self.Drop2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = grad_reverse(x)
        yhat = self.Drop1(self.D1(x))
        return torch.sigmoid(yhat)


