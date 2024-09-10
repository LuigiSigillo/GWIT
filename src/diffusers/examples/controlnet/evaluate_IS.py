import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from scipy.stats import entropy
import argparse

# python evaluate_IS.py --root /mnt/media/luigi/model_out_CVPR_MULTISUB_FIXED_CAPTION/controlnet/generated 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', type=str, default='picture-gene')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, choices=["cuda:0", "cpu"], default="cuda:0")
args = parser.parse_args()

# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def inception_score(batch_size=args.batch_size, resize=True):
    # Set up dtype
    device = torch.device(args.device)
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')

    files = readDir()
    N = len(files)
    preds = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))

    for i in tqdm(range(0, N, batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        y = get_pred(batch)
        preds[i:i + batch_size] = y

    assert batch_size > 0
    assert N > batch_size

    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    py = np.mean(preds, axis=0)  # marginal probability
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence

    mean_kl = np.mean(scores)
    inception_score = np.exp(mean_kl)

    return inception_score


def readDir(dirPath=args.root):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            f = os.path.join(dirPath, f)
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles.extend(subFiles)
            else:
                if "_gt" not in f:
                    allFiles.append(f)
        return allFiles
    else:
        return 'Error, not a dir'

if __name__ == '__main__':
    IS = inception_score()
    print('The Inception Score is %.4f' % IS)