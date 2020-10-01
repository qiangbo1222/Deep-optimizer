from __future__ import print_function
import numpy as np
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hid1 = nn.Linear(1024, 1024)
        self.hid1_drop = nn.Dropout(0.5)
        self.hid2 = nn.Linear(1024, 391)
        self.hid2_drop = nn.Dropout(0.5)
        self.out = nn.Linear(391,782)

    def forward(self,x):
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = F.relu(self.out(x))
        x = x.view(x.shape[0], 2, 391)
        x = F.softmax(x, dim=1)
        x = x.view(x.shape[0], 782)
        return x






