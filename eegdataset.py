import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
from torch.utils.data import Dataset

class EEGImageNet(Dataset):
    def __init__(self, root_dir, transform=None, resample=False):
        self.root_dir = root_dir
        self.transform = transform

        self.data = []
        for f in os.listdir(root_dir):
            eeg_sample = np.load(os.path.join(root_dir, f))
            # Discard first 20 seconds of EEG data and trim to 440 seconds
            self.data.append(eeg_sample[:, 20:20+440])

        print("Preprocessing EEG data...")
        # Compute and add the new channel
        min_data, max_data = np.min(self.data), np.max(self.data)
        new_channel = []
        for i in range(len(self.data)):
            seq = self.data[i]
            min_seq, max_seq = np.min(seq), np.max(seq)
            # Compute new channel value
            value = (max_seq - min_seq) / (max_data - min_data)
            new_channel.append([value]*440)
            # Normalize the sequence
            self.data[i] = (seq - min_seq) / (max_seq - min_seq)

        new_channel = np.array(new_channel).reshape(-1, 1, 440)
        self.data = np.concatenate((self.data, new_channel), axis=1) #(#, 20, 440)

        # Resample
        if resample:
            current_freq = 1000  # 1kHz
            target_freq = 256
            # up = 1
            # down = round(current_freq / target_freq)
            # self.resampled = scipy.signal.resample_poly(self.data[:, :-1], up, down, axis=-1)
            self.resampled = librosa.resample(self.data[:, :-1], orig_sr=current_freq, target_sr=target_freq, axis=-1)
            print(self.resampled.shape) #(11965, 19, 113)
            self.resampled = np.concatenate((self.resampled, new_channel[..., :self.resampled.shape[-1]]), axis=1)
            print(self.resampled.shape) #(11965, 20, 113)
            self.data = self.resampled
        # #dovrebbe essere cosi
        # self.data_len = 512
        print("Done!")
        print("Data shape:", self.data.shape)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = self.data[idx]
        if self.transform:
            eeg = self.transform(eeg)
        return eeg
    
    def plot_eeg(self, idx):
        eeg = self.data[idx]
        # eeg_resampled = self.resampled[idx]
        for i in range(eeg.shape[0]):
            plt.figure(figsize=(20, 5))
            # Plot the original data
            plt.subplot(1, 2, 1)
            plt.plot(eeg[i])
            plt.title(f'Original Channel {i+1}')
            # Plot the resampled data
            # plt.subplot(1, 2, 2)
            # plt.plot(eeg_resampled[i])
            # plt.title(f'Resampled Channel {i+1}')
            # plt.show()

if __name__ == "__main__":
    dataset = EEGImageNet('/home/luigi/Documents/DrEEam/dataset/data_10-20/mne_data', resample=True)
    print(len(dataset))
    dataset.plot_eeg(1)

import torch
def preprocess_EEG_data(batch_torch, resample=False):
    #take only the channels of interest 10-20
    # print("batch_torch shape: ", batch_torch.shape) 
    batch_torch = batch_torch[:,[1,2,3,4,5,6,7, 12, 13,14,15, 16, 23, 27, 24, 25, 26, 29, 31 ],:]
    # Discard first 20 seconds of EEG data and trim to 440 seconds
    # batch_torch = batch_torch[:, :, 20:20+440] #(#,19,519) --> (#,19,440) # non serve più visto che gigi l'ha messo in EEGDataset di dreamdiff
    # print("Preprocessing EEG data...")
    # Compute and add the new channel
    min_data, max_data = -60.032818, 56.274456
    new_channel = []
    for i,seq in enumerate(batch_torch):
        min_seq, max_seq = torch.min(seq), torch.max(seq)
        # Compute new channel value
        value = (max_seq - min_seq) / (max_data - min_data)
        new_channel.append(torch.stack([value]*440))
        # Normalize the sequence
        batch_torch[i] = (seq - min_seq) / (max_seq - min_seq)

    new_channel = torch.stack(new_channel).unsqueeze(1)
    data = torch.cat([batch_torch, new_channel], 1)

    # Resample to change in torch style
    if resample:
        current_freq = 1000  # 1kHz
        target_freq = 256
        resampled = librosa.resample(data[:, :-1].cpu().numpy(), orig_sr=current_freq, target_sr=target_freq, axis=-1)
        # print(resampled.shape) #(11965, 19, 113)
        resampled = np.concatenate((resampled, new_channel.cpu().numpy()[..., :resampled.shape[-1]]), axis=1)
        # print(resampled.shape) #(11965, 20, 113)
        data = torch.from_numpy(resampled).to("cuda").requires_grad_(data.requires_grad)
    # print(data.shape)
    return data