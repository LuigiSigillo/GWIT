# ## Code for making an EEG signal Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
import config
import random

torch.manual_seed(45)
np.random.seed(45)
random.seed(45)

class EEG_Encoder(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=128, num_layers=1):
        super(EEG_Encoder, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)

    def forward(self, x):

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device) 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device)

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)
        # x = F.normalize(x, dim=-1)
        # print(x.shape, feat.shape)
        return x#, feat