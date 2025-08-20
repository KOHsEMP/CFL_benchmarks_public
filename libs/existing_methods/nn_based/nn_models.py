import os
import sys
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

'''
Cite from: 
    https://github.com/takashiishida/comp/blob/master/models.py
'''
class mlp_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc_final(out)
        return out
    


class mlp_4layers(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(mlp_4layers, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.relu4 = nn.ReLU()
        self.fc_final = nn.Linear(hidden_dims[3], output_dim)
        

    def forward(self, x):
        #out = x.view(-1, self.num_flat_features(x))
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.relu4(self.fc4(out))
        out = self.fc_final(out)
        return out