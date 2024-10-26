import pickle
from tqdm import tqdm
import numpy as np
import torch
from dataset_EEG import EEGDatasetTVIZ
import librosa
# # train_path      = config.train_path
# # validation_path = config.validation_path
# device          = "cuda"
# thoughtviz_path ='/mnt/media/luigi/dataset/dreamdiff/tviz/eeg/image/data.pkl'
        
# with open(thoughtviz_path, 'rb') as file:
#     data = pickle.load(file, encoding='latin1')
#     train_X = data['x_train']
#     train_Y = data['y_train']
#     val_X = data['x_test']
#     val_Y = data['y_test']

# #load the data
# ## Training data
# x_train_eeg = []
# x_train_image = []
# labels = []
# x_train_subject=[]

# class_labels   = {}
# label_count    = 0

# for idx in tqdm(range(train_X.shape[0])):
#     # x_train_eeg.append(np.transpose(train_X[idx], (2, 1, 0)))
#     x_train_eeg.append(np.squeeze(np.transpose(train_X[idx], (2, 1, 0)), axis=0))
#     x_train_image.append(np.zeros(shape=(2, 2)))
#     x_train_subject.append(0)
#     labels.append(np.argmax(train_Y[idx]))
    
# x_train_eeg   = np.array(x_train_eeg)
# x_train_image = np.array(x_train_image)
# train_labels  = np.array(labels)
# x_train_subject = np.array(x_train_subject)

# print(x_train_eeg.shape, x_train_image.shape, train_labels.shape, x_train_subject.shape)
# print('Total number of classes: {}'.format(len(np.unique(train_labels))), np.unique(train_labels))

# # ## convert numpy array to tensor
# x_train_eeg   = torch.from_numpy(x_train_eeg).float()#.to(device)
# x_train_image = torch.from_numpy(x_train_image).float()#.to(device)
# train_labels  = torch.from_numpy(train_labels).long()#.to(device)
# x_train_subject  = torch.from_numpy(x_train_subject).long()#.to(device)

# train_data       = EEGDatasetTVIZ(x_train_eeg, x_train_image, train_labels, x_train_subject)
# #train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


# ## Validation data
# x_val_eeg   = []
# x_val_image = []
# label_val   = []
# x_val_subject = []

# for idx in tqdm(range(val_X.shape[0])):
#     # x_val_eeg.append(np.transpose(val_X[idx], (2, 1, 0)))
#     x_val_eeg.append(np.squeeze(np.transpose(val_X[idx], (2, 1, 0)), axis=0))
#     x_val_image.append(np.zeros(shape=(2, 2)))
#     x_val_subject.append(0.0)
#     label_val.append(np.argmax(val_Y[idx]))

# x_val_eeg   = np.array(x_val_eeg)
# x_val_image = np.array(x_val_image)
# label_val   = np.array(label_val)
# x_val_subject = np.array(x_val_subject)

# print(x_val_eeg.shape, x_val_image.shape, label_val.shape, x_val_subject.shape)
# print('Total number of classes: {}'.format(len(np.unique(label_val))), np.unique(label_val))

# x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
# x_val_image = torch.from_numpy(x_val_image).float()#.to(device)
# label_val   = torch.from_numpy(label_val).long().to(device)
# x_val_subject  = torch.from_numpy(x_val_subject).long()#.to(device)

# val_data       = EEGDatasetTVIZ(x_val_eeg, x_val_image, label_val, x_val_subject)
# # val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)
# print(val_data[0])


x_test_eeg,x_test_image,label_test, subject_test,label_folder_test = [],[],[], [], []
base_path       = '/mnt/media/luigi/dataset/dreamdiff/'
train_path      = 'tviz/thoughtviz/train/'
# validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'tviz/thoughtviz/test/'
from natsort import natsorted
import os
import cv2
from PIL import Image as PILIMAGE

datasets = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

class EEG2ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images = []
        self.labels = []
        self.subjects = []
        self.class_name = []
        self.eeg_no_resample = []
        self.image_shape = [3, 224,224]
        IMAGE_CLASSES = {'Apple': 0, 'Car': 1, 'Dog': 2, 'Gold': 3, 'Mobile': 4, 'Rose': 5, "Scooter": 6, 'Tiger': 7, 'Wallet': 8, 'Watch': 9}
        self.id_toclass = {v: k for k, v in IMAGE_CLASSES.items()}
        added_images = {}  # Dictionary to keep track of added images and their counters

        print('loading dataset...')
        for path in tqdm(natsorted(list_all_files(self.dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)
            # if loaded_array[2] in cls:
            upsampled_eeg_data = librosa.resample(loaded_array[0].squeeze(), orig_sr=1, target_sr=512 / 32, scale=True)

            num_repeats = 128 // upsampled_eeg_data.shape[0]  # Repeat 9 times
            extra_repeats = 128 % upsampled_eeg_data.shape[0]  # 2 additional channels needed
            # Replicate the channels and add extra channels
            replicated_eeg_data = np.tile(upsampled_eeg_data, (num_repeats, 1))
            extra_channels = upsampled_eeg_data[:extra_repeats, :]  # Take additional channels to make it 128
            final_eeg_data = np.vstack([replicated_eeg_data, extra_channels])
            
            eeg = np.float32(final_eeg_data)
            self.eegs.append(eeg)
            self.eeg_no_resample.append(loaded_array[0].squeeze())
            # self.eegs.append(np.expand_dims(loaded_array[1].T, axis=0))
            img = cv2.resize(loaded_array[2], (self.image_shape[1], self.image_shape[2]))
            # anoi serve IMG per Hugginface 
            #self.images.append(np.transpose(img, (2, 0, 1)))
            self.images.append(PILIMAGE.fromarray(img))
            self.labels.append(loaded_array[1])
            # Convert image to a hashable format (e.g., tuple of pixel values)
            img_hash = tuple(loaded_array[2].flatten())
            if img_hash in added_images:
                added_images[img_hash] += 1  # Increment counter for this image
                if added_images[img_hash] > 23:
                    added_images[img_hash] = 0

                    # raise Exception(f"Image {img_hash} has been added more than 24 times")
            else:
                added_images[img_hash] = 0  # Initialize counter for this image
            # self.class_name.append(loaded_array[3])
            self.subjects.append(added_images[img_hash])
            
        self.eegs     = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        #commented for Hugginface
        # self.images   = torch.from_numpy(np.array(self.images)).to(torch.float32)
        self.eeg_no_resample = torch.from_numpy(np.array(self.eeg_no_resample)).to(torch.float32)
        self.labels   = torch.from_numpy(np.array(self.labels)).to(torch.int32)
        self.class_name = torch.from_numpy(np.array(self.subjects)).to(torch.int32)


    def __len__(self):
        return self.eegs.shape[0]

    def __getitem__(self, idx):
        eeg   = self.eegs[idx]
        norm  = torch.max(eeg) / 2.0
        eeg   =  ( eeg - norm ) / norm
        image = self.images[idx]
        label = self.labels[idx]
        subject = self.subjects[idx]
        # con   = self.eeg_feat[idx]
        # class_n = self.class_name[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        # return eeg, image, label, con, class_n
        eeg_orig = self.eeg_no_resample[idx]
        norm  = torch.max(eeg_orig) / 2.0
        eeg_orig   =  ( eeg_orig - norm ) / norm
        return {'conditioning_image': eeg, 
                'caption': "image of a " + self.id_toclass[label.item()], #+ self.name_map[label.item()], 
                'image': image, #pil image
                'eeg_no_resample': eeg_orig, #eeg orig
                'label': label.item(),
                'subject': subject,
                }
        return image, con
    
    def get_label(self, idx):
        # label = self._get_raw_labels()[self._raw_idx[idx]]
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
        # label = self.labels[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        con = self.eeg_feat[idx]
        return con
    


from datasets import Dataset
from datasets import Dataset, Features, Array2D, Image, Value

features = Features({"image": Image(), 
                    "conditioning_image": Array2D(shape=(128, 512), dtype='float32'), 
                    "caption": Value("string"),
                    "eeg_no_resample": Array2D(shape=(14, 32), dtype='float32'), 
                    "label": Value("int32"),
                    "subject": Value("int32")
                    }
                    )

def gen_train():
    ## or if it's an IterableDataset
    for ex in dataset_train:
        if ex is None:
            continue
        yield ex
def gen_test():
    ## or if it's an IterableDataset
    for ex in dataset_test:
        if ex is None:
            continue
        yield ex

image_shape = [3, 224,224]
IMAGE_CLASSES = {'Apple': 0, 'Car': 1, 'Dog': 2, 'Gold': 3, 'Mobile': 4, 'Rose': 5, "Scooter": 6, 'Tiger': 7, 'Wallet': 8, 'Watch': 9}
id_toclass = {v: k for k, v in IMAGE_CLASSES.items()}
added_images = {}  # Dictionary to keep track of added images and their counters
dataset_path = base_path+train_path
def gen_train_at_the_fly():
    for path in tqdm(natsorted(list_all_files(dataset_path))):
        loaded_array = np.load(path, allow_pickle=True)
        # if loaded_array[2] in cls:
        upsampled_eeg_data = librosa.resample(loaded_array[0].squeeze(), orig_sr=1, target_sr=512 / 32, scale=True)

        num_repeats = 128 // upsampled_eeg_data.shape[0]  # Repeat 9 times
        extra_repeats = 128 % upsampled_eeg_data.shape[0]  # 2 additional channels needed
        # Replicate the channels and add extra channels
        replicated_eeg_data = np.tile(upsampled_eeg_data, (num_repeats, 1))
        extra_channels = upsampled_eeg_data[:extra_repeats, :]  # Take additional channels to make it 128
        final_eeg_data = np.vstack([replicated_eeg_data, extra_channels])
        
        eeg = np.float32(final_eeg_data)
        img = cv2.resize(loaded_array[2], (image_shape[1], image_shape[2]))
        # anoi serve IMG per Hugginface 
        #self.images.append(np.transpose(img, (2, 0, 1)))
        image = PILIMAGE.fromarray(img)
        label = loaded_array[1]
        # Convert image to a hashable format (e.g., tuple of pixel values)
        img_hash = tuple(loaded_array[2].flatten())
        if img_hash in added_images:
            added_images[img_hash] += 1  # Increment counter for this image
            if added_images[img_hash] > 23:
                added_images[img_hash] = 0

                # raise Exception(f"Image {img_hash} has been added more than 24 times")
        else:
            added_images[img_hash] = 0  # Initialize counter for this image
        # self.class_name.append(loaded_array[3])
        subject = added_images[img_hash]
        eeg = torch.from_numpy(np.array(eeg)).to(torch.float32)
        #commented for Hugginface
        # self.images   = torch.from_numpy(np.array(self.images)).to(torch.float32)
        # self.eeg_feat = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        label   = torch.from_numpy(np.array(label)).to(torch.int32)
        norm  = torch.max(eeg) / 2.0
        eeg   =  ( eeg - norm ) / norm
        # con   = self.eeg_feat[idx]
        # class_n = self.class_name[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        # return eeg, image, label, con, class_n
        eeg_orig = torch.from_numpy(np.array(loaded_array[0].squeeze())).to(torch.float32)
        norm  = torch.max(eeg_orig) / 2.0
        eeg_orig   =  ( eeg_orig - norm ) / norm
        
        yield {'conditioning_image': eeg, 
                'caption': "image of a " + id_toclass[label.item()], #+ self.name_map[label.item()], 
                'image': image, #pil image
                'eeg_no_resample': eeg_orig, #eeg orig
                'label': label.item(),
                'subject': subject,
                }

dataset_test = EEG2ImageDataset(base_path+test_path)
dset_test = Dataset.from_generator(gen_test, split='test', features=features, cache_dir="/mnt/media/luigi/dataset").with_format(type='torch')
dset_test.push_to_hub("luigi-s/EEG_Image_TVIZ_ALL_subj", private=True)

# dataset_train = EEG2ImageDataset(base_path+train_path)
dset_train = Dataset.from_generator(gen_train_at_the_fly, split='train',features=features, cache_dir="/mnt/media/luigi/dataset").with_format(type='torch')
dset_train.push_to_hub("luigi-s/EEG_Image_TVIZ_ALL_subj", private=True)


# dset_val = Dataset.from_generator(gen_val, split='validation', features=features).with_format(type='torch')
# dset_val.push_to_hub("luigi-s/EEG_Image_CVPR_ALL_subj", private=True)
