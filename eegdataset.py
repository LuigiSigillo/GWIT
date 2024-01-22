import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
from torch.utils.data import Dataset

class EEGImageNet(Dataset):
    def __init__(self, root_dir, transform=None, resample=True):
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
        self.data = np.concatenate((self.data, new_channel), axis=1)

        # Resample
        if resample:
            current_freq = 1000  # 1kHz
            target_freq = 256
            # up = 1
            # down = round(current_freq / target_freq)
            # self.resampled = scipy.signal.resample_poly(self.data[:, :-1], up, down, axis=-1)
            self.resampled = librosa.resample(self.data[:, :-1], orig_sr=current_freq, target_sr=target_freq, axis=-1)
            self.resampled = np.concatenate((self.resampled, new_channel[..., :self.resampled.shape[-1]]), axis=1)
            self.data = self.resampled
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
    dataset = EEGImageNet('/mnt/media/lopez/eeg_imagenet_10-20/mne_data', resample=False)
    print(len(dataset))
    dataset.plot_eeg(1)