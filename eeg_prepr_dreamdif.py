import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from pathlib import Path
from sklearn.model_selection import train_test_split

class EEG_preprocessing(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_len =  960
        self.data_chan =  128
        self.data = []
        self.input_paths = [str(f) for f in sorted(Path(root_dir).rglob('*'))]
        np.random.shuffle(self.input_paths)
        
        print("Preprocessing EEG data...")

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]
        eeg_tot = []
        data = np.load(data_path)
        # Adjust time series length
        #print(data.shape)
        if data.shape[-1] > self.data_len: #if too long choose a random integer in the range shape_vero - 512 e prendo da li i 512
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)   
            data = data[:, idx: idx+self.data_len]

        else:  # If too short, interpolate between the data
            x = np.linspace(0,  1, data.shape[-1])  
            x2 = np.linspace(0,  1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)

        ret = np.zeros((self.data_chan, self.data_len))

        # Adjust number of channels
        if self.data_chan > data.shape[-2]:  # If less channels than required
            for i in range(self.data_chan // data.shape[-2]):
                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data

            if self.data_chan % data.shape[-2] !=  0:
                ret[-(self.data_chan % data.shape[-2]):, :] = data[: (self.data_chan % data.shape[-2]), :]

        elif self.data_chan < data.shape[-2]:  # If more channels than required
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]

        elif self.data_chan == data.shape[-2]:  # If exactly the required number of channels
            #ret = data
            ret = data

        ret = ret /  10  # Reduce an order QUESTO NON HO CAPITO PERCHE' LO FA
        ret = torch.from_numpy(ret).float()
        ret = ret.unsqueeze(0)
        return ret
        #return {'eeg': ret}
        
if __name__ == "__main__":
    dataset = EEG_preprocessing(root_dir='/home/beingfedericax/moab3.9/dataset_eeg_final')
    print('Dataset creation completed!')
    
