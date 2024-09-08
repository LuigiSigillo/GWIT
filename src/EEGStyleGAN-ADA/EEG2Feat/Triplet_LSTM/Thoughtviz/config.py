import numpy as np
import torch
import os

np.random.seed(45)
torch.manual_seed(45)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



base_path       = '/path to data folder/'
thoughtviz_path = 'data/b2i_data/eeg/image/data.pkl'
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis_freq        = 1

# Hyper-parameters
feat_dim       = 128 # This will give 240 dim feature
projection_dim = 128
num_classes    = 10
input_size     = 14#128 # Number of EEG channels
timestep       = 32#440
# hidden_size    = embedding_dim//2
num_layers     = 4
batch_size     = 256 #48
test_batch_size = 256


temperature    = 0.5
epoch          = 4096
lr             = 3e-4

# Data Augmentation Hyper-parameters

max_shift    = 10
crop_size    = (timestep , 110)
noise_factor = 0.05
