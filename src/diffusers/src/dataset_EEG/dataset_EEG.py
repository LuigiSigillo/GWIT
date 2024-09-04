import torch
from torch.utils.data import Dataset
def identity(x):
    return x
import numpy as np
from transformers import AutoProcessor
import os 
from scipy.interpolate import interp1d
from PIL import Image
import json

with open('/home/luigi/Documents/DrEEam/src/diffusers/src/dataset_EEG/imagenet-simple-labels.json') as f:
    labels = json.load(f)

def class_id_to_label(i):
    return labels[i]

class EEGDataset(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=identity, subject=4, 
                 encoder_name = 'bendr', imagenet_path = '/mnt/media/luigi/dataset/imageNet_images/',
                   only_eeg=False):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        self.subject = subject
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']       
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = imagenet_path
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        self.only_eeg = only_eeg
        # Compute size
        self.size = len(self.data)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.data_l = [torch.tensor(item["eeg"])[:,20:460] for item in self.data]
        # # pad all tensors to have same length
        # self.data_l = torch.nn.utils.rnn.pad_sequence(self.data_l, batch_first=True)
        # stack them
        # self.data_l = self.data_l.permute(0,2,1)
        self.mean = torch.mean(torch.stack(self.data_l))
        self.std = torch.std(torch.stack(self.data_l))
        del self.data_l
        self.encoder_name = encoder_name
    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        # print(self.data[i])
        # eeg = self.data[i]["eeg"].float().t()
        if self.encoder_name == 'bendr':
            eeg = self.data[i]["eeg"].float()
            eeg = eeg[:,20:460]
            # print(eeg.shape)
        ##### 2023 2 13 add preprocess and transpose
            ### lopex noon la vuole piu
            # eeg = np.array(eeg.transpose(0,1))
            # x = np.linspace(0, 1, eeg.shape[-1])
            # x2 = np.linspace(0, 1, self.data_len)
            # f = interp1d(x, eeg)
            # eeg = f(x2)
            # eeg = torch.from_numpy(eeg).float()
            
            #normalize eeg
            eeg = (eeg - self.mean) / self.std
        else:
            eeg = self.data[i]["eeg"].float().t()

            eeg = eeg[20:460,:]
            ##### 2023 2 13 add preprocess and transpose
            eeg = np.array(eeg.transpose(0,1))
            x = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, eeg)
            eeg = f(x2)
            eeg = torch.from_numpy(eeg).float()
        if self.only_eeg:
            return {'eeg': eeg}
        
        ##### 2023 2 13 add preprocess
        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        # print(image_path)
        try:
            image_raw = Image.open(image_path).convert('RGB') 
        except:
            return None #use collate fn to filter the none
        image = np.array(image_raw) / 255.0
        #probably need to change this
        image = image_raw
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        # return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}
        return {'conditioning_image': eeg, 
                'caption': "image of a " + class_id_to_label(image_name.split('_')[0]), 
                'image': image,
                'label_folder': image_name.split('_')[0],
                'subject': self.data[i]["subject"],
                }
        # return eeg, label




class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # from sklearn.model_selection import train_test_split
        # train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.8)
        # self.split_idx =  train_idx if split_name=="train" else test_idx

        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        # self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return len(self.split_idx)

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]

    def __add__(self, other):
        if not isinstance(other, Splitter):
            return NotImplemented
        
        # Define how to combine two Splitter objects
        self.dataset = self.dataset + other.dataset
        # combined_split_path = self.split_path + other.split_path
        # combined_split_num = self.split_num + other.split_num
        # combined_split_name = self.split_name + "_" + other.split_name
        # combined_subject = self.subject + other.subject
        self.split_idx = self.split_idx + other.split_idx
        
        return self
