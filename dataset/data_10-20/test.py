import torch 
import numpy as np
import os
import json 

file = torch.load('/home/luigi/Documents/eeg/eeg_5_95_std.pth')
print(file.keys())
dataset = file['dataset']
# splits = file['splits']
print(len((dataset)))

listona = []
ten_twenty_indices = [1,2,3,4,5,6,7, 12, 13,14,15, 16, 23, 27, 24, 25, 26, 29, 31 ]

for i in range((len(dataset))):
    dizi = {}
    elem = dataset[i]
    eeg_tensor = elem['eeg'][ten_twenty_indices]
    eeg_array = eeg_tensor.numpy()
    index = i
    dizi['label'] = elem['label']
    dizi['image'] = elem['image']
    dizi['subject'] = elem['subject']
    dizi['path'] = os.path.join('/home/luigi/Documents/eeg/mne_data', str(index))
    listona.append(dizi)
    np.save(os.path.join('/home/luigi/Documents/eeg/mne_data', str(index)), eeg_array)

with open('/home/luigi/Documents/eeg/metadata.json', 'w') as f:
    json.dump(listona, f)