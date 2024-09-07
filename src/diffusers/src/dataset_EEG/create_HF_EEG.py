# from dataset_EEG import EEGDataset, Splitter
# import torchvision.transforms as transforms
# import torch
# from einops import rearrange
# from datasets import Dataset
# from datasets import Dataset, Features, Array2D, Image, Value

# def normalize(img):
#     if img.shape[-1] == 3:
#         img = rearrange(img, 'h w c -> c h w')
#     img = torch.tensor(img)
#     img = img * 2.0 - 1.0 # to -1 ~ 1
#     return img

# class random_crop:
#     def __init__(self, size, p):
#         self.size = size
#         self.p = p
#     def __call__(self, img):
#         if torch.rand(1) < self.p:
#             return transforms.RandomCrop(size=(self.size, self.size))(img)
#         return img


# def channel_last(img):
#     if img.shape[-1] == 3:
#         return img
#     return rearrange(img, 'c h w -> h w c')

# eeg_signals_path = '/mnt/media/luigi/dataset/dreamdiff/eeg_5_95_std.pth'
# splits_path = '/mnt/media/luigi/dataset/dreamdiff/block_splits_by_image_all.pth'
# imagenet_path = '/mnt/media/luigi/dataset/dreamdiff/imageNet_images/'

# crop_ratio = 0.2
# img_size = 512
# crop_pix = int(crop_ratio*img_size)
# encoder_name = 'loro' 
# only_eeg = False
# # subject = 4
# img_transform_train = transforms.Compose([
#     normalize,
#     transforms.Resize((512, 512)),
#     random_crop(img_size-crop_pix, p=0.5),
#     transforms.Resize((512, 512)),
#     channel_last
# ])
# img_transform_test = transforms.Compose([
#     normalize, 
#     transforms.Resize((512, 512)),
#     channel_last
# ])

# image_transform = [img_transform_train, img_transform_test]

# features = Features({"image": Image(), 
#                     "conditioning_image": Array2D(shape=(128, 512), dtype='float32'), 
#                     "caption": Value("string"),
#                     "label_folder": Value("string"),
#                     "subject": Value("int32")
#                     }
#                     )

# dataset_train_l, dataset_test_l, dataset_val_l = [], [], []

# for subject in range(6):
#     dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject, encoder_name, imagenet_path=imagenet_path, only_eeg=only_eeg)
#     dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)
#     dataset_val = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)

#     split_train = Splitter(dataset_train, split_path=splits_path, split_num=0, split_name='train', subject=subject)
#     split_test = Splitter(dataset_test, split_path=splits_path, split_num=0, split_name='test', subject=subject)
#     split_val = Splitter(dataset_val, split_path=splits_path, split_num=0, split_name='val', subject=subject)
    
#     dataset_train_l.append(split_train)
#     dataset_test_l.append(split_test)
#     dataset_val_l.append(split_val)

# subject = 0 # 0 = ALL
# dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject, encoder_name, imagenet_path=imagenet_path, only_eeg=only_eeg)
# dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)
# dataset_val = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)

# split_train = Splitter(dataset_train, split_path=splits_path, split_num=0, split_name='train', subject=subject)
# split_test = Splitter(dataset_test, split_path=splits_path, split_num=0, split_name='test', subject=subject)
# split_val = Splitter(dataset_val, split_path=splits_path, split_num=0, split_name='val', subject=subject)


# def gen_train():
#     ## or if it's an IterableDataset
#     for ex in split_train:
#         if ex is None:
#             continue
#         yield ex
# def gen_test():
#     ## or if it's an IterableDataset
#     for ex in split_test:
#         if ex is None:
#             continue
#         yield ex
# def gen_val():
#     ## or if it's an IterableDataset
#     for ex in split_val:
#         if ex is None:
#             continue
#         yield ex

# dataset_train_l = dataset_train_l[0] + dataset_train_l[1] + dataset_train_l[2] + dataset_train_l[3] + dataset_train_l[4] + dataset_train_l[5]
# dataset_test_l = dataset_test_l[0] + dataset_test_l[1] + dataset_test_l[2] + dataset_test_l[3] + dataset_test_l[4] + dataset_test_l[5]
# dataset_val_l = dataset_val_l[0] + dataset_val_l[1] + dataset_val_l[2] + dataset_val_l[3] + dataset_val_l[4] + dataset_val_l[5]


# dset_train = Dataset.from_generator(gen_train, split='train',features=features).with_format(type='torch')
# dset_train.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

# dset_test = Dataset.from_generator(gen_test, split='test', features=features).with_format(type='torch')
# dset_test.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

# dset_val = Dataset.from_generator(gen_val, split='validation', features=features).with_format(type='torch')
# dset_val.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

# print("TIPI dataset classico")
# # print(type(dataset_val[20]['conditioning_image']))
# # print(type(dataset_val[0]['image']))
# print(dataset_val[20]['caption'], dataset_val[20]['label_folder'])
# print(dataset_val[10]['caption'], dataset_val[10]['label_folder'])

# # print(type(dataset_val[0]['subject']))

# print("TIPI dataset HF")
# # print(type(dset_val[0]['conditioning_image']))
# # print(type(dset_val[0]['image']))
# print(dset_val[20]['caption'], dset_val[20]['label_folder'])
# print(dset_val[10]['caption'], dset_val[10]['label_folder'])

# # print(type(dset_val[0]['subject']))


# from datasets import load_dataset
# data = load_dataset('luigi-s/EEG_Image_ALL_subj', split='train')
# data = load_dataset('luigi-s/EEG_Image_ALL_subj', split='validation')
# data = load_dataset('luigi-s/EEG_Image_ALL_subj', split='test')


# for d in data:
#     # print(d['caption'], d['label_folder'])
#     print(d['subject'])

import cv2
from natsort import natsorted
import os
import numpy as np
import torch
from tqdm import tqdm
from dataset_EEG import EEGDatasetCVPR

x_test_eeg,x_test_image,label_test, subject_test,label_folder_test = [],[],[], [], []
base_path       = '/mnt/media/luigi/dataset/dreamdiff/'
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'eeg_imagenet40_cvpr_2017_raw/test/'

datasets = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for path in [validation_path, test_path]:
    x_test_eeg,x_test_image,label_test, subject_test,label_folder_test = [],[],[], [], []

    for i in tqdm(natsorted(os.listdir(base_path + path))):
        loaded_array = np.load(base_path + path + i, allow_pickle=True)
        x_test_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_test_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # 	test_cluster += 1
        # label_test.append(class_labels[loaded_array[3]])
        label_test.append(loaded_array[2])
        label_folder_test.append(loaded_array[3])
        subject_test.append(loaded_array[4])
        
    x_test_eeg   = np.array(x_test_eeg)
    x_test_image = np.array(x_test_image)
    test_labels  = np.array(label_test)
    subject_test = np.array(subject_test)
    # label_folder_test = np.array(label_folder_test)

    x_test_eeg   = torch.from_numpy(x_test_eeg).float()#.to(device)
    x_test_image = torch.from_numpy(x_test_image).float()#.to(device)
    test_labels  = torch.from_numpy(test_labels).long()#.to(device)
    subject_test = torch.from_numpy(subject_test).long()#.to(device)
    # label_folder_test = torch.from_numpy(label_folder_test).to(device)

    test_data       = EEGDatasetCVPR(x_test_eeg, x_test_image, test_labels, subject_test, label_folder_test)
    datasets.append(test_data)


# dataset_train, dataset_val, dataset_test = datasets
dataset_val, dataset_test = datasets[0], datasets[1]
from datasets import Dataset
from datasets import Dataset, Features, Array2D, Image, Value

features = Features({"image": Image(), 
                    "conditioning_image": Array2D(shape=(128, 440), dtype='float32'), 
                    "caption": Value("string"),
                    "label_folder": Value("string"),
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
def gen_val():
    ## or if it's an IterableDataset
    for ex in dataset_val:
        if ex is None:
            continue
        yield ex


# dset_train = Dataset.from_generator(gen_train, split='train',features=features).with_format(type='torch')
# dset_train.push_to_hub("luigi-s/EEG_Image_CVPR_ALL_subj", private=True)

dset_test = Dataset.from_generator(gen_test, split='test', features=features).with_format(type='torch')
dset_test.push_to_hub("luigi-s/EEG_Image_CVPR_ALL_subj", private=True)

dset_val = Dataset.from_generator(gen_val, split='validation', features=features).with_format(type='torch')
dset_val.push_to_hub("luigi-s/EEG_Image_CVPR_ALL_subj", private=True)

# print("TIPI dataset classico")
# # print(type(dataset_val[20]['conditioning_image']))
# print(type(dataset_test[0]['image']))
# print(dataset_test[20]['caption'], dataset_test[20]['label_folder'])
# print(dataset_test[10]['caption'], dataset_test[10]['label_folder'])

# # print(type(dataset_val[0]['subject']))

# print("TIPI dataset HF")
# # print(type(dset_val[0]['conditioning_image']))
# print(type(dset_test[0]['image']))
# print(dset_test[20]['caption'], dset_test[20]['label_folder'])
# print(dset_test[10]['caption'], dset_test[10]['label_folder'])

# # # print(type(dset_val[0]['subject']))
