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
import gc

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
batch_size      = config.batch_size
embedding_dim   = config.embedding_dim
projection_dim  = config.projection_dim
num_layers      = config.num_layers
epochs          = config.epoch

# load the dataset
x_train_eeg = []
x_train_image = []
labels = []

for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_train_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_train_image.append(img)
    labels.append(loaded_array[2])

# convert to torch tensors
x_train_eeg = np.array(x_train_eeg)
x_train_image = np.array(x_train_image)
labels = np.array(labels)

x_train_eeg = torch.from_numpy(x_train_eeg).float()
x_train_image = torch.from_numpy(x_train_image).float()
labels = torch.from_numpy(labels).long()

train_data = torch.utils.data.TensorDataset(x_train_eeg, x_train_image, labels)
data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)

### Validation dataset
x_val_eeg = []
x_val_image = []
label_Val = []

for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
    loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_val_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_val_image.append(img)
    label_Val.append(loaded_array[2])

x_val_eeg = np.array(x_val_eeg)
x_val_image = np.array(x_val_image)
labels_val = np.array(label_Val)

x_val_eeg = torch.from_numpy(x_val_eeg).float().to(device)
x_val_image = torch.from_numpy(x_val_image).float().to(device)
labels_val = torch.from_numpy(labels_val).long().to(device)

model_path = 'EEGClip_ckpt/EXPERIMENT_2/checkpoints/'
print(natsorted(os.listdir(model_path)))

for i in reversed(natsorted(os.listdir(model_path))):

    print("#########################################################################################")
    print('Model: ', i)

    model_path_c = model_path + i

    checkpoint = torch.load(model_path_c, map_location=device)
    eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)

    image_embedding = resnet50(pretrained=False).to(device)
    num_features = image_embedding.fc.in_features

    image_embedding.fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_features, config.embedding_dim, bias=False)
    )

    image_embedding.fc.to(device)

    model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)

    model = model.text_encoder

    for param in model.parameters():
        param.requires_grad = True

    new_layer = nn.Sequential(
        nn.Linear(config.embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 40),
        nn.Softmax(dim=1)
    )

    model.fc = nn.Sequential(
        model.fc,
        new_layer
    )

    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def train(model, data_loader, optimizer, criterion, device, epoch):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs_eeg, inputs_image, labels = data
            inputs_eeg, inputs_image, labels = inputs_eeg.to(device), inputs_image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_eeg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / i+1
        print('[%d] loss: %.3f' % (epoch + 1, loss))
        return loss

    def evaluate(model):
        model.eval()
        outputs = model(x_val_eeg)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels_val).sum().item()
        print('Accuracy of the network %d %%' % (100 * correct / 1994))
        val_acc = 100 * correct / 1994
        return val_acc

    dir_info  = natsorted(glob('FineTuningEEG/EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    if not os.path.isdir('FineTuningEEG/EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('FineTuningEEG/EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.system('cp *.py FineTuningEEG/EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('FineTuningEEG/EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('FineTuningEEG/EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    best_val_acc   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, epochs):

        running_train_loss = train(model, data_loader, optimizer, criterion, device, epoch)
        val_acc   = evaluate(model)


        if best_val_acc < val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'FineTuningEEG/EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, best_val_epoch, val_acc))
    del model
    torch.cuda.empty_cache()
    gc.collect()
