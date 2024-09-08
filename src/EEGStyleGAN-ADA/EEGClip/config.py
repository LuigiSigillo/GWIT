import numpy as np
import torch
import os

np.random.seed(45)
torch.manual_seed(45)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



base_path       = '../data/'
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'eeg_imagenet40_cvpr_2017_raw/test/'

# train_path      = 'eeg_imagenet40_cvpr_2017_5_45hz/train/'
# validation_path = 'eeg_imagenet40_cvpr_2017_5_45hz/val/'
# test_path       = 'eeg_imagenet40_cvpr_2017_5_45hz/test/'

device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# Hyper-parameters
embedding_dim  = 256
projection_dim = 256
input_channels = 128 # Number of EEG channels
hidden_size    = embedding_dim//2
num_layers     = 1
batch_size     = 64
test_batch_size= 64
epoch          = 2049
# epoch          = 251 # for EEG fine-tuning
# epoch          = 10 # for image fine-tuning
temperature    = 0.5
