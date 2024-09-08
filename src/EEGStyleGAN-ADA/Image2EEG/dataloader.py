import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import config
import cv2
from dataaugmentation import apply_augmentation, extract_freq_band
from torchvision.models import AlexNet_Weights, ResNet18_Weights, GoogLeNet_Weights

class EEGDataset(Dataset):
    def __init__(self, eegs, images, labels, subject_num=None, avgeeg_feat=None):
        self.eegs         = eegs
        self.images       = images
        self.labels       = labels
        self.avgeeg_feat  = avgeeg_feat
        self.subject_num  = subject_num
        # self.preprocess   = AlexNet_Weights.IMAGENET1K_V1.transforms()
        # self.preprocess   = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.preprocess   = GoogLeNet_Weights.IMAGENET1K_V1.transforms()


    def __getitem__(self, index):
        eeg    = self.eegs[index]
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        image  = self.preprocess(self.images[index])
        label  = self.labels[index]
        subject= self.subject_num[index]
        con    = np.zeros(shape=(config.n_subjects,), dtype=np.float32)
        con[subject.numpy()-1] = 1.0
        con   = torch.from_numpy(con)

        if self.avgeeg_feat != None:
            avgeeg = self.avgeeg_feat[index]
            return eeg, image, label, con, avgeeg
        else:
            return eeg, image, label, con

    def __len__(self):
        return len(self.eegs)



if __name__ == '__main__':

    base_path       = config.base_path
    train_path      = config.train_path
    validation_path = config.validation_path
    device          = config.device
    print(device)

    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        labels.append(loaded_array[2])

    x_train_eeg = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    labels = np.array(labels)

    # ## convert numpy array to tensor
    x_train_eeg = torch.from_numpy(x_train_eeg).float().to(device)
    x_train_image = torch.from_numpy(x_train_image).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    train_data  = EEGTrainDataset(x_train_eeg, x_train_image, labels)