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
test_path = config.test_path
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
                
            EEG_embedding        = np.concatenate((EEG_embedding, EEG_feat.cpu().detach().numpy()), axis=0) if EEG_embedding.size else EEG_feat.cpu().detach().numpy()
            image_embeddings      = np.concatenate((image_embeddings, image_feat.cpu().detach().numpy()), axis=0) if image_embeddings.size else image_feat.cpu().detach().numpy()
            EEG_embedding_proj   = np.concatenate((EEG_embedding_proj, EEG_embed.cpu().detach().numpy()), axis=0) if EEG_embedding_proj.size else EEG_embed.cpu().detach().numpy()
            image_embeddings_proj = np.concatenate((image_embeddings_proj, image_embed.cpu().detach().numpy()), axis=0) if image_embeddings_proj.size else image_embed.cpu().detach().numpy()
            labels_array          = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
            
            # print(EEG_embedding[:2],image_embeddings[:2])
        # scheduler.step()
        pd.DataFrame(EEG_embedding_proj).to_csv("/home/brainimage/brain2image/code_v7/EEG_embed_train.csv")
        pd.DataFrame(image_embeddings_proj).to_csv("/home/brainimage/brain2image/code_v7/image_embed_train.csv")
            
            
        if epoch % 5 == 0:
        ### compute k-means score on the EEG and image embeddings
            num_clusters = 40
            k_means = K_means(n_clusters=num_clusters)
            (clustered_EEG_label, clustered_image_label), (EEG_score, image_score) = k_means.transform(EEG_embedding, image_embeddings, labels_array, labels_array)
            k_means_proj = K_means(n_clusters=num_clusters)
            (clustered_EEG_label, clustered_image_label), (EEG_score_proj, image_score_proj) = k_means_proj.transform(EEG_embedding_proj, image_embeddings_proj, labels_array, labels_array)
            print(f"EEG KMeans score{epoch}:", EEG_score)
            print(f"Image KMeans score{epoch}:", image_score)
            print(f"EEG KMeans score proj{epoch}:", EEG_score_proj)
            print(f"Image KMeans score proj{epoch}:", image_score_proj)
            # print(clustered_EEG_label)
        
        if epoch == 32 or epoch == 64 or epoch == 128 or epoch == 256 or epoch == 512:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, 'EXPERIMENT_{}/checkpoints/clip_{}.pth'.format(experiment_num, epoch)) 
        
def validate(preprocess, model, temperature, experiment_num):
    EEG_embedding = np.array([])
    image_embeddings = np.array([])
    EEG_embedding_proj = np.array([])
    image_embeddings_proj = np.array([])
    labels_array = np.array([])
    logsoftmax   = nn.LogSoftmax(dim=-1)
    
    with torch.no_grad():
        total_loss = 0.0        
        for batch_idx, (EEGs, images, labels) in enumerate(tqdm(val_data_loader)):
            # get the embeddings for the EEG and images
            EEGs, images, labels = EEGs.to(device), images.to(device), labels.to(device)
            images = preprocess(images)
            EEG_embed, image_embed, EEG_feat, image_feat = model(EEGs, images)
            
            logits = (EEG_embed @ image_embed.T) / temperature
            images_similarity = image_embed @ image_embed.T
            EEGs_similarity = EEG_embed @ EEG_embed.T
            
            targets = F.softmax(
                (images_similarity + EEGs_similarity) / 2 * temperature, dim=-1
            )
            # EEGs_loss = F.cross_entropy(logits, targets, reduction='none')
            # images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
            # loss =  (images_loss + EEGs_loss) / 2.0
            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            EEGs_loss  = cross_entropy(logits, targets, reduction='none')
            loss        = (images_loss + EEGs_loss) / 2.0
            total_loss += loss.mean().item()
            
            EEG_embedding = np.concatenate((EEG_embedding, EEG_feat.cpu().numpy()), axis=0) if EEG_embedding.size else EEG_feat.cpu().numpy()
            image_embeddings = np.concatenate((image_embeddings, image_feat.cpu().numpy()), axis=0) if image_embeddings.size else image_feat.cpu().numpy()
            EEG_embedding_proj   = np.concatenate((EEG_embedding_proj, EEG_embed.cpu().detach().numpy()), axis=0) if EEG_embedding_proj.size else EEG_embed.cpu().detach().numpy()
            image_embeddings_proj = np.concatenate((image_embeddings_proj, image_embed.cpu().detach().numpy()), axis=0) if image_embeddings_proj.size else image_embed.cpu().detach().numpy()
            labels_array = np.concatenate((labels_array, labels.cpu().numpy()), axis=0) if labels_array.size else labels.cpu().numpy()
            
        pd.DataFrame(EEG_embedding_proj).to_csv("/home/brainimage/brain2image/code_v7/EEG_embed.csv")
        pd.DataFrame(image_embeddings_proj).to_csv("/home/brainimage/brain2image/code_v7/image_embed.csv")
            
        print('Validation loss: {}'.format(total_loss / len(val_data_loader)))
        
    ### compute k-means score and Umap score on the EEG and image embeddings
    num_clusters = 40
    k_means = K_means(n_clusters=num_clusters)
    (clustered_EEG_label, clustered_image_label), (EEG_score, image_score) = k_means.transform(EEG_embedding, image_embeddings, labels_array, labels_array)
    k_means_proj = K_means(n_clusters=num_clusters)
    (clustered_EEG_label, clustered_image_label), (EEG_score_proj, image_score_proj) = k_means_proj.transform(EEG_embedding_proj, image_embeddings_proj, labels_array, labels_array)
    print(f"EEG KMeans score{epoch}:", EEG_score)
    print(f"Image KMeans score{epoch}:", image_score)
    print(f"EEG KMeans score proj{epoch}:", EEG_score_proj)
    print(f"Image KMeans score proj{epoch}:", image_score_proj)
    print(clustered_EEG_label)
        

## Validation data
x_test_eeg = []
x_test_image = []
label_test = []

for i in tqdm(natsorted(os.listdir(base_path + test_path))):
    loaded_array = np.load(base_path + test_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_test_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = (np.float32(np.transpose(img, (2, 0, 1)))-127.5) / 127.5
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_test_image.append(img)
    label_test.append(loaded_array[2])
    
x_test_eeg = np.array(x_test_eeg)
x_test_image = np.array(x_test_image)
labels_test = np.array(label_test)
# x_test_eeg = (x_test_eeg - norm_mean) / norm_std
# print(np.min(x_test_eeg), np.max(x_test_eeg), np.min(x_test_image), np.max(x_test_image))
# x_test_eeg = (x_test_eeg - norm_min) / (norm_max - norm_min)
# print(np.min(x_test_eeg), np.max(x_test_eeg))
# ## hyperparameters
input_channels = config.input_channels
embedding_dim  = config.embedding_dim
projection_dim = config.projection_dim
input_size     = config.input_channels
hidden_size    = config.hidden_size
num_layers     = config.num_layers
batch_size     = config.batch_size
epoch          = config.epoch


x_test_eeg = torch.from_numpy(x_test_eeg).float().to(device)
x_test_image = torch.from_numpy(x_test_image).float().to(device)
labels_test = torch.from_numpy(labels_test).long().to(device)
print(x_test_eeg.shape, x_test_image.shape, labels_test.shape)

test_data = torch.utils.data.TensorDataset(x_test_eeg, x_test_image, labels_test)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=True)

# ## define the model
eeg_embedding = EEG_Encoder().to(device)
# image_embedding = ModifiedResNet(layers = [3, 4, 6, 3], output_dim=embedding_dim, heads=8).to(device)

image_embedding = torchvision.models.resnet50(pretrained=True).to(device)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms().to(device)

for param in image_embedding.parameters():
    param.requires_grad = True
num_features = image_embedding.fc.in_features
## add relu
image_embedding.fc = nn.Sequential(
    nn.ReLU(),
    nn.Linear(num_features, config.embedding_dim, bias=False)
)
image_embedding.fc.to(device)
# pdb.set_trace()
model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=0.003)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_image, T_max=1024, eta_min=3e-4)
scheduler = None
# dir_info  = natsorted(glob('EXPERIMENT_*'))
# if len(dir_info)==0:
#     experiment_num = 1
# else:
#     experiment_num = int(dir_info[-1].split('_')[-1]) + 1
experiment_num = 57

if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
    os.makedirs('EXPERIMENT_{}'.format(experiment_num))
    os.makedirs('EXPERIMENT_{}/umap'.format(experiment_num))
    os.makedirs('EXPERIMENT_{}/tsne'.format(experiment_num))
    os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/clip_*.pth'.format(experiment_num)))

START_EPOCH = 0

if len(ckpt_lst)>=1:
    ckpt_path  = ckpt_lst[-1]
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    START_EPOCH += 1
else:
    os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))


# train(preprocess, model, 0.5, optimizer, scheduler, START_EPOCH, epoch, experiment_num)

# validate(preprocess, model, 0.5, experiment_num)

print('completed')