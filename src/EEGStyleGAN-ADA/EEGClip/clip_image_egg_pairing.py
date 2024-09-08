import config
from tqdm import tqdm
import numpy as np
import pdb
import os
from natsort import natsorted
import cv2
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import math

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
input_channels = config.input_channels
embedding_dim  = config.embedding_dim
projection_dim = config.projection_dim
input_size     = config.input_channels
hidden_size    = config.hidden_size
num_layers     = config.num_layers
batch_size     = config.batch_size
epoch          = config.epoch


model_path = '/home/brainimage/brain2image/code_v9/EXPERIMENT_105/checkpoints/'
fmodel_path =  natsorted(os.listdir(model_path))[-1]
print(fmodel_path)

checkpoint = torch.load(model_path+fmodel_path, map_location=device)
eeg_embedding = EEG_Encoder().to(device)

image_embedding = resnet50(pretrained=False).to(device)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms().to(device)
num_features = image_embedding.fc.in_features
image_embedding.fc = nn.Sequential(
    nn.ReLU(),
    nn.Linear(num_features, config.embedding_dim, bias=False)
)
image_embedding.fc.to(device)

model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(config.device)

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


# Get unique elements and their indices from x_val_image
_, unique_indices = np.unique(x_val_image, return_index=True, axis=0)

# Remove non-unique elements
x_val_image_unique = x_val_image[unique_indices]
x_val_eeg_unique = x_val_eeg[unique_indices]
labels_val_unique = labels_val[unique_indices]

x_val_eeg = x_val_eeg_unique
x_val_image = x_val_image_unique
labels_val = labels_val_unique

print(x_val_eeg.shape, x_val_image.shape, labels_val.shape)

x_val_eeg = torch.from_numpy(x_val_eeg).float().to(device)
x_val_image = torch.from_numpy(x_val_image).float().to(device)
labels_val = torch.from_numpy(labels_val).long().to(device)

val_data = torch.utils.data.TensorDataset(x_val_eeg, x_val_image, labels_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle=False)

def find_matches(model, image_embeddings, query, image_filenames, number, n=9):

    text_embeddings_n = model.text_encoder(query)
    text_embeddings_n = text_embeddings_n.detach().cpu()
    dot_similarity = text_embeddings_n @ image_embeddings.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    # print(indices.shape)
    matches = [image_filenames[idx] for idx in indices]
    # pdb.set_trace()
    # _, axes = plt.subplots(int(math.sqrt(n)), int(math.sqrt(n)), figsize=(10, 10))
    # for match, ax in zip(matches, axes.flatten()):
    #     # image = cv2.imread(f"{CFG.image_path}/{match}")
    #     image_array = match.cpu().numpy()
    #     image_array = np.transpose(image_array, (1, 2, 0))
    #     image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    #     ax.imshow(image)
    #     ax.axis("off")
    
    # plt.savefig(f"Matche/matches+{number}.png")

    # Calculate the number of rows and columns for the grid

    grid_rows = int(math.sqrt(n))
    grid_cols = int(math.sqrt(n))

    # Create an empty canvas to hold the grid
    grid_height = grid_rows * 224  # Replace `image_height` with the actual height of your images
    grid_width = grid_cols * 224    # Replace `image_width` with the actual width of your images
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Iterate over the images and place them in the grid
    for i, match in enumerate(matches):
        image_array = match.cpu().numpy()
        image_array = np.transpose(image_array, (1, 2, 0))
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Calculate the row and column index for the current image
        row = i // grid_cols
        col = i % grid_cols

        # Calculate the top-left corner coordinates for placing the image
        top = row * 224  
        left = col * 224

        # Place the image on the grid canvas
        grid_canvas[top:top+224, left:left+224] = image*255

    # Save the grid image
    cv2.imwrite(f"Matches/matchess+{number}.png", grid_canvas)


def find_matches2(model, image_embeddings, query, image_filenames, number, n=9):
    text_embeddings_n = model.text_encoder(query)
    text_embeddings_n = text_embeddings_n.detach().cpu()
    dot_similarity = text_embeddings_n @ image_embeddings.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices]

    # Calculate the number of rows and columns for the grid
    grid_rows = int(math.sqrt(n))
    grid_cols = int(math.sqrt(n))

    # Create a folder to save the images
    folder_name = f"Matches/matches{number}"
    os.makedirs(folder_name, exist_ok=True)

    # Create an empty canvas to hold the grid
    grid_height = grid_rows * 224  # Replace `image_height` with the actual height of your images
    grid_width = grid_cols * 224    # Replace `image_width` with the actual width of your images
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Iterate over the images and place them in the grid
    for i, match in enumerate(matches):
        image_array = match.cpu().numpy()
        image_array = np.transpose(image_array, (1, 2, 0))
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Calculate the row and column index for the current image
        row = i // grid_cols
        col = i % grid_cols

        # Calculate the top-left corner coordinates for placing the image
        top = row * 224  
        left = col * 224

        # Place the image on the grid canvas
        grid_canvas[top:top+224, left:left+224] = image*255

        # Save the image
        image_path = os.path.join(folder_name, f"match_{i+1}.png")
        cv2.imwrite(image_path, image*255)

    # Save the grid image
    grid_image_path = os.path.join(folder_name, f"matches.png")
    cv2.imwrite(grid_image_path, grid_canvas)

    return grid_image_path


def find_matches5(model, image_embeddings, query, image_filenames, number, n=10):
    text_embeddings_n = model.text_encoder(query)
    text_embeddings_n = text_embeddings_n.detach().cpu()
    dot_similarity = text_embeddings_n @ image_embeddings.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices]

    # Create a folder to save the images
    folder_name = f"Matches/matches{number}"
    
    grid_rows = 1
    grid_cols = n

    # Create an empty canvas to hold the grid
    grid_height = 224  # Replace `image_height` with the actual height of your images
    grid_width = grid_cols * 224    # Replace `image_width` with the actual width of your images
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Iterate over the images and place them in the grid
    for i, match in enumerate(matches):
        image_array = match.cpu().numpy()
        image_array = np.transpose(image_array, (1, 2, 0))
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Calculate the column index for the current image
        col = i

        # Calculate the top-left corner coordinates for placing the image
        top = 0
        left = col * 224

        # Place the image on the grid canvas
        grid_canvas[top:top+224, left:left+224] = image*255

        # Save the image
        image_path = os.path.join(folder_name, f"match_{i+1}.png")
        cv2.imwrite(image_path, image*255)

    # Save the grid image
    grid_image_path = os.path.join(folder_name, f"matches.png")
    cv2.imwrite(grid_image_path, grid_canvas)

    return grid_image_path



total_image_embedding = torch.zeros((len(val_loader),128)).to('cpu')
for i, (eeg, image, label) in enumerate(tqdm(val_loader)):
    image = preprocess(image)
    image_embeddings = model.image_encoder(image)
    total_image_embedding[i, :] = image_embeddings.squeeze(0).detach().cpu()

# unique_tensor = torch.unique(total_image_embedding, dim=0)
# print(unique_tensor.shape)

for i, (eeg, image, label) in enumerate(tqdm(val_loader)):
    query = eeg # torch.Size([1, 440, 128])
    image_expected = np.transpose(image[0].cpu().numpy(), (1, 2, 0))
    image_expected = cv2.cvtColor(image_expected, cv2.COLOR_RGB2BGR)
    folder_name = f"Matches/matches{i}"
    os.makedirs(folder_name, exist_ok=True)
    cv2.imwrite(f"Matches/matches{i}/expected+{i}.png", image_expected*255)
    # x_val_image = x_val_image.unique(dim=0)
    # pdb.set_trace()
    find_matches5(model, total_image_embedding, query, x_val_image.detach().cpu(), i)
        
