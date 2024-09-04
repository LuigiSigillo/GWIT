from dataset_EEG import EEGDataset, Splitter
import torchvision.transforms as transforms
import torch
from einops import rearrange
from datasets import Dataset
from datasets import Dataset, Features, Array2D, Image, Value

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

eeg_signals_path = '/mnt/media/luigi/dataset/dreamdiff/eeg_5_95_std.pth'
splits_path = '/mnt/media/luigi/dataset/dreamdiff/block_splits_by_image_all.pth'
imagenet_path = '/mnt/media/luigi/dataset/dreamdiff/imageNet_images/'

crop_ratio = 0.2
img_size = 512
crop_pix = int(crop_ratio*img_size)
encoder_name = 'loro' 
only_eeg = False
# subject = 4
img_transform_train = transforms.Compose([
    normalize,
    transforms.Resize((512, 512)),
    random_crop(img_size-crop_pix, p=0.5),
    transforms.Resize((512, 512)),
    channel_last
])
img_transform_test = transforms.Compose([
    normalize, 
    transforms.Resize((512, 512)),
    channel_last
])

image_transform = [img_transform_train, img_transform_test]

features = Features({"image": Image(), 
                    "conditioning_image": Array2D(shape=(128, 512), dtype='float32'), 
                    "caption": Value("string"),
                    "label_folder": Value("string"),
                    "subject": Value("int32")
                    }
                    )

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

subject = 0 # 0 = ALL
dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject, encoder_name, imagenet_path=imagenet_path, only_eeg=only_eeg)
dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)
dataset_val = EEGDataset(eeg_signals_path, image_transform[1], subject, encoder_name,  imagenet_path=imagenet_path, only_eeg=only_eeg)

split_train = Splitter(dataset_train, split_path=splits_path, split_num=0, split_name='train', subject=subject)
split_test = Splitter(dataset_test, split_path=splits_path, split_num=0, split_name='test', subject=subject)
split_val = Splitter(dataset_val, split_path=splits_path, split_num=0, split_name='val', subject=subject)


def gen_train():
    ## or if it's an IterableDataset
    for ex in split_train:
        if ex is None:
            continue
        yield ex
def gen_test():
    ## or if it's an IterableDataset
    for ex in split_test:
        if ex is None:
            continue
        yield ex
def gen_val():
    ## or if it's an IterableDataset
    for ex in split_val:
        if ex is None:
            continue
        yield ex

# dataset_train_l = dataset_train_l[0] + dataset_train_l[1] + dataset_train_l[2] + dataset_train_l[3] + dataset_train_l[4] + dataset_train_l[5]
# dataset_test_l = dataset_test_l[0] + dataset_test_l[1] + dataset_test_l[2] + dataset_test_l[3] + dataset_test_l[4] + dataset_test_l[5]
# dataset_val_l = dataset_val_l[0] + dataset_val_l[1] + dataset_val_l[2] + dataset_val_l[3] + dataset_val_l[4] + dataset_val_l[5]


dset_train = Dataset.from_generator(gen_train, split='train',features=features).with_format(type='torch')
dset_train.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

dset_test = Dataset.from_generator(gen_test, split='test', features=features).with_format(type='torch')
dset_test.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

dset_val = Dataset.from_generator(gen_val, split='validation', features=features).with_format(type='torch')
dset_val.push_to_hub("luigi-s/EEG_Image_ALL_subj", private=True)

print("TIPI dataset classico")
# print(type(dataset_val[20]['conditioning_image']))
# print(type(dataset_val[0]['image']))
print(dataset_val[20]['caption'], dataset_val[20]['label_folder'])
print(dataset_val[10]['caption'], dataset_val[10]['label_folder'])

# print(type(dataset_val[0]['subject']))

print("TIPI dataset HF")
# print(type(dset_val[0]['conditioning_image']))
# print(type(dset_val[0]['image']))
print(dset_val[20]['caption'], dset_val[20]['label_folder'])
print(dset_val[10]['caption'], dset_val[10]['label_folder'])

# print(type(dset_val[0]['subject']))



