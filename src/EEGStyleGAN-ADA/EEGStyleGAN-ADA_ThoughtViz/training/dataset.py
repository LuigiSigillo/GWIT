# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import cv2
import config
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from torch.utils.data import Dataset
from network import EEGFeatNet, ImageFeatNet, EEGCNNFeatNet
# from torchvision.models import GoogLeNet_Weights
import torchvision.transforms as transforms
from visualizations import K_means

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self,
#         name,                   # Name of the dataset.
#         raw_shape,              # Shape of the raw image data (NCHW).
#         max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
#         use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
#         xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
#         random_seed = 0,        # Random seed to use when applying max_size.
#     ):
#         self._name = name
#         self._raw_shape = list(raw_shape)
#         self._use_labels = use_labels
#         self._raw_labels = None
#         self._label_shape = None

#         # Apply max_size.
#         self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
#         if (max_size is not None) and (self._raw_idx.size > max_size):
#             np.random.RandomState(random_seed).shuffle(self._raw_idx)
#             self._raw_idx = np.sort(self._raw_idx[:max_size])

#         # Apply xflip.
#         self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
#         if xflip:
#             self._raw_idx = np.tile(self._raw_idx, 2)
#             self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

#     def _get_raw_labels(self):
#         if self._raw_labels is None:
#             self._raw_labels = self._load_raw_labels() if self._use_labels else None
#             if self._raw_labels is None:
#                 self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
#             assert isinstance(self._raw_labels, np.ndarray)
#             assert self._raw_labels.shape[0] == self._raw_shape[0]
#             assert self._raw_labels.dtype in [np.float32, np.int64]
#             if self._raw_labels.dtype == np.int64:
#                 assert self._raw_labels.ndim == 1
#                 assert np.all(self._raw_labels >= 0)
#         return self._raw_labels

#     def close(self): # to be overridden by subclass
#         pass

#     def _load_raw_image(self, raw_idx): # to be overridden by subclass
#         raise NotImplementedError

#     def _load_raw_labels(self): # to be overridden by subclass
#         raise NotImplementedError

#     def __getstate__(self):
#         return dict(self.__dict__, _raw_labels=None)

#     def __del__(self):
#         try:
#             self.close()
#         except:
#             pass

#     def __len__(self):
#         return self._raw_idx.size

#     def __getitem__(self, idx):
#         image = self._load_raw_image(self._raw_idx[idx])
#         assert isinstance(image, np.ndarray)
#         assert list(image.shape) == self.image_shape
#         assert image.dtype == np.uint8
#         if self._xflip[idx]:
#             assert image.ndim == 3 # CHW
#             image = image[:, :, ::-1]
#         return image.copy(), self.get_label(idx)

#     def get_label(self, idx):
#         label = self._get_raw_labels()[self._raw_idx[idx]]
#         if label.dtype == np.int64:
#             onehot = np.zeros(self.label_shape, dtype=np.float32)
#             onehot[label] = 1
#             label = onehot
#         return label.copy()

#     def get_details(self, idx):
#         d = dnnlib.EasyDict()
#         d.raw_idx = int(self._raw_idx[idx])
#         d.xflip = (int(self._xflip[idx]) != 0)
#         d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
#         return d

#     @property
#     def name(self):
#         return self._name

#     @property
#     def image_shape(self):
#         return list(self._raw_shape[1:])

#     @property
#     def num_channels(self):
#         assert len(self.image_shape) == 3 # CHW
#         return self.image_shape[0]

#     @property
#     def resolution(self):
#         assert len(self.image_shape) == 3 # CHW
#         assert self.image_shape[1] == self.image_shape[2]
#         return self.image_shape[1]

#     @property
#     def label_shape(self):
#         if self._label_shape is None:
#             raw_labels = self._get_raw_labels()
#             if raw_labels.dtype == np.int64:
#                 self._label_shape = [int(np.max(raw_labels)) + 1]
#             else:
#                 self._label_shape = raw_labels.shape[1:]
#         return list(self._label_shape)

#     @property
#     def label_dim(self):
#         assert len(self.label_shape) == 1
#         return self.label_shape[0]

#     @property
#     def has_labels(self):
#         return any(x != 0 for x in self.label_shape)

#     @property
#     def has_onehot_labels(self):
#         return self._get_raw_labels().dtype == np.int64

# #----------------------------------------------------------------------------

# class ImageFolderDataset(Dataset):
#     def __init__(self,
#         path,                   # Path to directory or zip.
#         resolution      = None, # Ensure specific resolution, None = highest available.
#         **super_kwargs,         # Additional arguments for the Dataset base class.
#     ):
#         self._path = path
#         self._zipfile = None

#         if os.path.isdir(self._path):
#             self._type = 'dir'
#             self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
#         elif self._file_ext(self._path) == '.zip':
#             self._type = 'zip'
#             self._all_fnames = set(self._get_zipfile().namelist())
#         else:
#             raise IOError('Path must point to a directory or zip')

#         PIL.Image.init()
#         self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
#         if len(self._image_fnames) == 0:
#             raise IOError('No image files found in the specified path')

#         name = os.path.splitext(os.path.basename(self._path))[0]
#         raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
#         if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
#             raise IOError('Image files do not match the specified resolution')
#         super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

#     @staticmethod
#     def _file_ext(fname):
#         return os.path.splitext(fname)[1].lower()

#     def _get_zipfile(self):
#         assert self._type == 'zip'
#         if self._zipfile is None:
#             self._zipfile = zipfile.ZipFile(self._path)
#         return self._zipfile

#     def _open_file(self, fname):
#         if self._type == 'dir':
#             return open(os.path.join(self._path, fname), 'rb')
#         if self._type == 'zip':
#             return self._get_zipfile().open(fname, 'r')
#         return None

#     def close(self):
#         try:
#             if self._zipfile is not None:
#                 self._zipfile.close()
#         finally:
#             self._zipfile = None

#     def __getstate__(self):
#         return dict(super().__getstate__(), _zipfile=None)

#     def _load_raw_image(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         with self._open_file(fname) as f:
#             if pyspng is not None and self._file_ext(fname) == '.png':
#                 image = pyspng.load(f.read())
#             else:
#                 image = np.array(PIL.Image.open(f))
#         if image.ndim == 2:
#             image = image[:, :, np.newaxis] # HW => HWC
#         image = image.transpose(2, 0, 1) # HWC => CHW
#         return image

#     def _load_raw_labels(self):
#         fname = 'dataset.json'
#         if fname not in self._all_fnames:
#             return None
#         with self._open_file(fname) as f:
#             labels = json.load(f)['labels']
#         if labels is None:
#             return None
#         labels = dict(labels)
#         labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
#         labels = np.array(labels)
#         labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
#         return labels

# #----------------------------------------------------------------------------

class EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images = []
        self.labels = []
        self.class_name = []
        self.eeg_feat = []
        cls_lst = [0, 1]
        self._raw_shape = [3, config.image_height, config.image_width]
        self.resolution = config.image_height
        self.has_labels  = True
        self.label_shape = [config.projection_dim]
        self.label_dim   = config.projection_dim
        self.name        = config.dataset_name
        self.image_shape = [3, config.image_height, config.image_width]
        self.num_channels = config.input_channel
        is_cnn = config.is_cnn

        ## Loading Pre-trained EEG Encoder #####
        self.eeg_model = EEGFeatNet(in_channels=config.input_size, n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
        self.eeg_model = torch.nn.DataParallel(self.eeg_model).to(config.device)
        eegckpt   = 'eegbestckpt/eegfeat_all_0.7212357954545454.pth'
        eegcheckpoint = torch.load(eegckpt, map_location=config.device)
        self.eeg_model.load_state_dict(eegcheckpoint['model_state_dict'])
        print('Loading EEG checkpoint from previous epoch: {}'.format(eegcheckpoint['epoch']))
        print('loading dataset...')
        for path in tqdm(natsorted(glob(self.dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)
            # if loaded_array[2] in cls:
            eeg = np.float32(np.squeeze(loaded_array[0], axis=-1).T)
            self.eegs.append(eeg)
            # self.eegs.append(np.expand_dims(loaded_array[1].T, axis=0))
            img = np.float32(cv2.resize(loaded_array[2], (config.image_height, config.image_width)))
            self.images.append(np.transpose(img, (2, 0, 1)))
            self.labels.append(loaded_array[1])
            # self.class_name.append(loaded_array[3])
            with torch.no_grad():
                norm = np.max(eeg) / 2.0
                eeg  = (eeg - norm) / norm
                self.eeg_feat.append(self.eeg_model(torch.from_numpy(np.expand_dims(eeg, axis=0)).to(config.device)).detach().cpu().numpy()[0])

        k_means             = K_means(n_clusters=config.n_classes)
        clustering_acc_proj = k_means.transform(np.array(self.eeg_feat), np.array(self.labels))
        print("[KMeans score Proj: {}]".format(clustering_acc_proj))

        self.eegs     = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        self.images   = torch.from_numpy(np.array(self.images)).to(torch.float32)
        self.eeg_feat = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        self.labels   = torch.from_numpy(np.array(self.labels)).to(torch.int32)
        # self.class_name = torch.from_numpy(np.array(self.class_name))


    def __len__(self):
        return self.eegs.shape[0]

    def __getitem__(self, idx):
        eeg   = self.eegs[idx]
        norm  = torch.max(eeg) / 2.0
        eeg   =  ( eeg - norm ) / norm
        image = self.images[idx]
        label = self.labels[idx]
        con   = self.eeg_feat[idx]
        # class_n = self.class_name[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        # return eeg, image, label, con, class_n
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

class Image2EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images = []
        self.labels = []
        self.class_name = []
        self.eeg_feat = np.array([])
        temp_images   = []
        self.subject_num = []
        cls_lst = [0, 1]
        self._raw_shape = [3, config.image_height, config.image_width]
        self.resolution = config.image_height
        self.has_labels  = True
        self.label_shape = [config.projection_dim]
        self.label_dim   = config.projection_dim
        self.name        = config.dataset_name
        self.image_shape = [3, config.image_height, config.image_width]
        self.num_channels = config.input_channel
        # self.preprocess   = GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        self.preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])

        # ## Loading Pre-trained EEG Encoder #####
        # self.eeg_model = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
        # self.eeg_model = torch.nn.DataParallel(self.eeg_model).to(config.device)
        # eegckpt   = 'eegbestckpt/eegfeat_all_0.9665178571428571.pth'
        # eegcheckpoint = torch.load(eegckpt, map_location=config.device)
        # self.eeg_model.load_state_dict(eegcheckpoint['model_state_dict'])
        # print('Loading EEG checkpoint from previous epoch: {}'.format(eegcheckpoint['epoch']))
        # ########################################

        ## Loading Pre-trained Image Encoder #####
        self.image_model     = ImageFeatNet(projection_dim=config.projection_dim).to(config.device)
        self.image_model     = torch.nn.DataParallel(self.image_model).to(config.device)
        ckpt_path = 'imageckpt/eegfeat_all_0.6875.pth'
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        self.image_model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading Image checkpoint from previous epoch: {}'.format(checkpoint['epoch']))
        ########################################

        print('loading dataset...')
        for path in tqdm(natsorted(glob(self.dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)
            # if loaded_array[2] in cls:
            self.eegs.append(loaded_array[1].T)
            # self.eegs.append(np.expand_dims(loaded_array[1].T, axis=0))
            self.images.append(np.transpose(np.float32(cv2.resize(loaded_array[0], (config.image_height, config.image_width))), (2, 0, 1)))
            self.labels.append(loaded_array[2])
            self.class_name.append(loaded_array[3])
            self.subject_num.append(loaded_array[4]) # Subject Number

            img = np.float32(loaded_array[0])
            img = self.preprocess( img ).numpy()
            c   = np.zeros(shape=(config.n_subjects,), dtype=np.float32)
            c[loaded_array[4]-1] = 1.0
            c 	= np.expand_dims( np.expand_dims(c, axis=-1), axis=-1 )
            c 	= np.tile(c, (1, img.shape[1], img.shape[2]))
            img = np.concatenate([img, c], axis=0)
            temp_images.append(img)
        
        temp_images = torch.from_numpy(np.array(temp_images)).to(torch.float32)
        for idx in tqdm(range(0, temp_images.shape[0], 256)):
            batch_images = temp_images[idx:idx+256].to(config.device)
            with torch.no_grad():
                feat  = (self.image_model(batch_images)).detach().cpu().numpy()
            self.eeg_feat = np.concatenate((self.eeg_feat, feat), axis=0) if self.eeg_feat.size else feat
        
        print(self.eeg_feat.shape)
        
        k_means             = K_means(n_clusters=config.n_classes)
        clustering_acc_proj = k_means.transform(np.array(self.eeg_feat), np.array(self.labels))
        print("[Test KMeans score Proj: {}]".format(clustering_acc_proj))

        self.eegs        = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        self.images      = torch.from_numpy(np.array(self.images)).to(torch.float32)
        self.eeg_feat    = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        self.labels      = torch.from_numpy(np.array(self.labels)).to(torch.int32)
        self.subject_num = torch.from_numpy(np.array(self.subject_num)).to(torch.int32)
        # self.class_name = torch.from_numpy(np.array(self.class_name))


    def __len__(self):
        return self.eegs.shape[0]

    def __getitem__(self, idx):
        eeg   = self.eegs[idx]
        norm  = torch.max(eeg) / 2.0
        eeg   =  ( eeg - norm ) / norm
        image = self.images[idx]
        label = self.labels[idx]
        con   = self.eeg_feat[idx]
        # class_n = self.class_name[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        # return eeg, image, label, con, class_n
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