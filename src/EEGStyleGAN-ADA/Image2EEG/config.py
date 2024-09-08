import numpy as np
import torch

np.random.seed(45)
torch.manual_seed(45)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



base_path       = '../../../data/'
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'eeg_imagenet40_cvpr_2017_raw/test/'

# base_path       = '/media/A/'
# train_path      = 'dataset/eeg_imagenet40_cvpr_2017_raw/train/'
# validation_path = 'dataset/eeg_imagenet40_cvpr_2017_raw/val/'
# test_path       = 'dataset/eeg_imagenet40_cvpr_2017_raw/test/'


device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis_freq        = 1

# Hyper-parameters
feat_dim       = 128 # This will give 240 dim feature
projection_dim = 128
num_classes    = 40
input_size     = 128 # Number of EEG channels
timestep       = 440
input_shape    = (1, 440, 128)
image_shape    = (3, 224, 224)
# hidden_size    = embedding_dim//2
num_layers     = 4
batch_size     = 256 #48
temperature    = 0.5
epoch          = 4096
lr             = 3e-4
n_subjects     = 6

# Data Augmentation Hyper-parameters

max_shift    = 10
crop_size    = (timestep , 110)
noise_factor = 0.05
