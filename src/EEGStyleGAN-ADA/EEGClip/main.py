## Take input of EEG and save it as a numpy array
import config
from tqdm import tqdm
import numpy as np
import pdb
import os
from natsort import natsorted
import cv2
from glob import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from EEG_encoder import EEG_Encoder
from model import ModifiedResNet
from CLIPModel import CLIPModel
from Visualizations import Umap, K_means, TsnePlot
from dataloader import CustomDataLoader, CustomDataset
from torchvision.models import resnet50, ResNet50_Weights


base_path       = config.base_path
train_path      = config.train_path
validation_path = config.validation_path
device          = config.device
print(device)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train(preprocess, model, temperature, optimizer, scheduler, START_EPOCH, num_epochs, experiment_num):
    
    for epoch in range(START_EPOCH, num_epochs):
        running_loss = 0.0
        EEG_embedding = np.array([])
        image_embeddings = np.array([])
        EEG_embedding_proj = np.array([])
        image_embeddings_proj = np.array([])
        labels_array = np.array([])
        logsoftmax   = nn.LogSoftmax(dim=-1)

        tq = tqdm(data_loader)
        for batch_idx, (EEGs, images, labels) in enumerate(tq):
            EEGs, images, labels = EEGs.to(device), images.to(device), labels.to(device)
            images = preprocess(images)
            # get the embeddings for the EEG and images
            optimizer.zero_grad()
            
            EEG_embed, image_embed, EEG_feat, image_feat = model(EEGs, images) 

            logits = (EEG_embed @ image_embed.T) * torch.exp(torch.tensor(temperature))

            labels = torch.arange(image_embed.shape[0]).to(config.device)
            
            loss_i = F.cross_entropy(logits, labels, reduction='none')
            loss_t = F.cross_entropy(logits.T, labels, reduction='none')

            loss = (loss_i + loss_t) / 2.0

            # pdb.set_trace()
            loss        = loss.mean() # average the loss over the batch
            
            # backpropagate and update parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            tq.set_description('[%d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, running_loss / (batch_idx+1.0)))
                
        if epoch == 32 or epoch == 64 or epoch == 128 or epoch == 256 or epoch == 512 or epoch == 1024 or epoch == 1536 or epoch == 2048 or epoch == 2560 or epoch == 3072 or epoch == 3584 or epoch == 4096:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, 'EEGClip_ckpt/EXPERIMENT_{}/checkpoints/clip_{}.pth'.format(experiment_num, epoch)) 

        
#load the data
## Training data
x_train_eeg = []
x_train_image = []
labels = []

for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    x_train_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_train_image.append(img)
    labels.append(loaded_array[2])
    
x_train_eeg = np.array(x_train_eeg)
x_train_image = np.array(x_train_image)
labels = np.array(labels)

## Validation data
x_val_eeg = []
x_val_image = []
label_Val = []

for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
    loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    x_val_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_val_image.append(img)
    label_Val.append(loaded_array[2])
    
x_val_eeg = np.array(x_val_eeg)
x_val_image = np.array(x_val_image)
labels_val = np.array(label_Val)

# ## hyperparameters
input_channels = config.input_channels
embedding_dim  = config.embedding_dim
projection_dim = config.projection_dim
input_size     = config.input_channels
hidden_size    = config.hidden_size
num_layers     = config.num_layers
batch_size     = config.batch_size
epoch          = config.epoch

# ## convert numpy array to tensor
x_train_eeg = torch.from_numpy(x_train_eeg).float().to(device)
x_train_image = torch.from_numpy(x_train_image).float().to(device)
labels = torch.from_numpy(labels).long().to(device)


# train_data  = CustomDataset(x_train_eeg, x_train_image, labels)

x_val_eeg = torch.from_numpy(x_val_eeg).float().to(device)
x_val_image = torch.from_numpy(x_val_image).float().to(device)
labels_val = torch.from_numpy(labels_val).long().to(device)
print(x_train_eeg.shape, x_train_image.shape, labels.shape, x_val_eeg.shape, x_val_image.shape, labels_val.shape)

train_data = torch.utils.data.TensorDataset(x_train_eeg, x_train_image, labels)
data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val_eeg, x_val_image, labels_val)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=True)

# ## define the model
eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)
image_embedding = torchvision.models.resnet50(pretrained=True).to(device)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms().to(device)

for param in image_embedding.parameters():
    param.requires_grad = False

num_features = image_embedding.fc.in_features

image_embedding.fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_features, config.embedding_dim, bias=False)
    )

image_embedding.fc.to(device)
model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=0.003)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_image, T_max=1024, eta_min=3e-4)
scheduler = None
dir_info  = natsorted(glob('EEGClip_ckpt/EXPERIMENT_*'))
if len(dir_info)==0:
    experiment_num = 1
else:
    experiment_num = int(dir_info[-1].split('_')[-1])  + 1

if not os.path.isdir('EEGClip_ckpt/EXPERIMENT_{}'.format(experiment_num)):
    os.makedirs('EEGClip_ckpt/EXPERIMENT_{}'.format(experiment_num))
    os.makedirs('EEGClip_ckpt/EXPERIMENT_{}/umap'.format(experiment_num))
    os.makedirs('EEGClip_ckpt/EXPERIMENT_{}/tsne'.format(experiment_num))
    os.system('cp *.py EEGClip_ckpt/EXPERIMENT_{}'.format(experiment_num))

ckpt_lst = natsorted(glob('EEGClip_ckpt/EXPERIMENT_{}/checkpoints/clip_*.pth'.format(experiment_num)))
print(experiment_num)
START_EPOCH = 0

if len(ckpt_lst)>=1:
    ckpt_path  = ckpt_lst[-1]
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    START_EPOCH += 1
else:
    os.makedirs('EEGClip_ckpt/EXPERIMENT_{}/checkpoints/'.format(experiment_num))

train(preprocess, model, 0.5, optimizer, scheduler, START_EPOCH, epoch, experiment_num)

print('completed')