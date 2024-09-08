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


class EEGDataset(Dataset):
    def __init__(self, eegs, images, labels, subjects=None, n_fft=64, win_length=64, hop_length=16):
        self.eegs         = eegs
        self.images       = images
        self.labels       = labels
        self.subjects     = subjects


    def __getitem__(self, index):
        eeg    = self.eegs[index]
        # eeg    = np.float32(self.eegs[index].cpu())
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        # eeg    = (eeg - np.min(eeg))/ (np.max(eeg) - np.min(eeg))
        image  = self.images[index]
        label  = self.labels[index]
        subject= self.subjects[index]
        # con    = np.zeros(shape=(config.n_subjects,), dtype=np.float32)
        # con[subject.numpy()-1] = 1.0
        # con   = torch.from_numpy(con)
        # eeg_x1 = np.float32(apply_augmentation(eeg, 'random_noise', max_shift=config.max_shift, crop_size=config.crop_size, noise_factor=config.noise_factor))
        # eeg_x2 = np.float32(apply_augmentation(eeg, 'random_noise', max_shift=config.max_shift, crop_size=config.crop_size, noise_factor=config.noise_factor))
        # gamma_band = np.float32(extract_freq_band(eeg, fs=1000, nperseg=440))
        # eeg    = np.float32(np.expand_dims(eeg, axis=0))
        # eeg_x1 = np.float32(np.expand_dims(eeg_x1, axis=0))
        # eeg_x2 = np.float32(np.expand_dims(eeg_x2, axis=0))
        # spectrogram = self.spectrograms[index]
        # return eeg, eeg_x1, eeg_x2, gamma_band, image, label
        # return eeg, eeg_x1, eeg_x2, image, label
        # return eeg, image, label, con
        return eeg, image, label, subject

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