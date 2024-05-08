import json
import os
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm

class MOABB(Dataset):
    def __init__(self, root_dir='moabb', in_channels=128, data_len=960, corrupted_files='dataset/moabb/corrupted_files.json', load_preprocessed=False, redo_preprocessing=False):
        self.root_dir = root_dir
        self.data_chan = in_channels
        self.data_len = data_len
        self.load_preprocessed = load_preprocessed

        files_dir = os.path.join(root_dir, "dataset_eeg_final")
        self.input_paths = [str(f) for f in sorted(Path(files_dir).rglob('*'))]
        
        # Exclude corrupted files
        with open(corrupted_files, 'r') as f:
            corrupted_files = json.load(f)['files']
        corrupted_files = [os.path.join(files_dir, file) for file in corrupted_files]
        self.input_paths = [path for path in self.input_paths if path not in corrupted_files]

        # Compute the subject stats for normalization
        if os.path.exists(os.path.join(root_dir, 'subject_stats.pkl')):
            with open(os.path.join(root_dir, 'subject_stats.pkl'), 'rb') as f:
                self.subject_stats = pickle.load(f)
                print("Subject stats loaded!")
        else:
            self.subject_stats = dict()
            subject_data = []
            current_dataset = self.input_paths[0].split("/")[-1].split("_")[0]
            current_subject = self.input_paths[0].split("/")[-1].split("_")[2]
            for path in tqdm(self.input_paths, desc="Calculating subject stats"):
                dataset_name = path.split("/")[-1].split("_")[0]
                subject_number = path.split("/")[-1].split("_")[2]
                if subject_number != current_subject or current_dataset != dataset_name:
                    self.subject_stats[(current_dataset, current_subject)] = {'mean': np.mean(subject_data), 'std': np.std(subject_data)}
                    subject_data = []
                    current_subject = subject_number
                    current_dataset = dataset_name
                subject_data.append(np.load(path))
            with open(os.path.join(root_dir, 'subject_stats.pkl'), 'wb') as f:
                pickle.dump(self.subject_stats, f)
            print("Subject stats saved!")
        
        # Preprocess the data
        preprocessed_data_dir = os.path.join(root_dir, "preprocessed_data")
        if self.load_preprocessed:
            if not os.path.exists(preprocessed_data_dir):
                os.makedirs(preprocessed_data_dir)
            if redo_preprocessing or not any(os.scandir(preprocessed_data_dir)):
                for path in tqdm(self.input_paths, desc="Preprocessing"):
                    data = np.load(path)
                    dataset_name = path.split("/")[-1].split("_")[0]
                    subject_number = path.split("/")[-1].split("_")[2]
                    data = self.preprocess(data, dataset_name, subject_number)
                    np.save(os.path.join(preprocessed_data_dir, path.split("/")[-1]), data)
            print("Using preprocessed data!")
            self.input_paths = [str(f) for f in sorted(Path(preprocessed_data_dir).rglob('*'))]

    def preprocess(self, data, dataset_name, subject_number):
        # Normalize
        data = (data - self.subject_stats[(dataset_name, subject_number)]['mean']) / self.subject_stats[(dataset_name, subject_number)]['std']
        
        # Adjust time series length
        if data.shape[-1] > self.data_len: #if too long choose a random integer in the range shape_vero - 512 e prendo da li i 512
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)   
            data = data[:, idx: idx+self.data_len]
        # else:  # If too short, interpolate between the data
        #     x = np.linspace(0,  1, data.shape[-1])  
        #     x2 = np.linspace(0,  1, self.data_len)
        #     f = interp1d(x, data)
        #     data = f(x2)
        
        # Adjust number of channels
        ret = np.zeros((self.data_chan, self.data_len))
        if self.data_chan > data.shape[-2]: # If less channels than required
            for i in range(self.data_chan // data.shape[-2]):
                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] !=  0:
                ret[-(self.data_chan % data.shape[-2]):, :] = data[: (self.data_chan % data.shape[-2]), :]
        elif self.data_chan < data.shape[-2]: # If more channels than required
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        elif self.data_chan == data.shape[-2]: # If exactly the required number of channels
            ret = data
        ret = ret /  10
        return ret

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]
        data = np.load(data_path)
        
        if not self.load_preprocessed:
            dataset_name = data_path.split("/")[-1].split("_")[0]
            subject_number = data_path.split("/")[-1].split("_")[2]
            data = self.preprocess(data, dataset_name, subject_number)
        elif self.data_len < data.shape[-1]:
            data = data[:, :self.data_len]

        data = torch.from_numpy(data).float()
        return [data] # return list because the dataloader expects a list of tensors where the first element is the batch input
        
if __name__ == "__main__":
    dataset = MOABB(root_dir='/mnt/media/lopez/moabb')
    print('Dataset creation completed!')
    
