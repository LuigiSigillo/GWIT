## Take input of EEG and save it as a numpy array
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
from tqdm import tqdm
import numpy as np
import pdb
import os
from natsort import natsorted
import cv2
from glob import glob
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import EEGDataset
from network import EEGFeatNet, ImageFeatNet
# from model import ModifiedResNet
# from CLIPModel import CLIPModel
from visualizations import Umap, K_means, TsnePlot, save_image
from losses import ContrastiveLoss
from dataaugmentation import apply_augmentation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox
from image3dplot import ImageAnnotations3D
from scipy.special import softmax


np.random.seed(45)
torch.manual_seed(45)


def calculate_recall_at_k(softmax_output, labels, k):
    """
    Calculate recall@k, which is the proportion of correctly classified items in the top k predicted classes.

    Parameters:
    softmax_output (ndarray): The output of the softmax layer, of shape (num_samples, num_classes).
    labels (ndarray): The true class labels, of shape (num_samples,).
    k (int): The number of top predicted classes to consider.

    Returns:
    float: The recall@k score, which is the proportion of correctly classified items in the top k predicted classes.
    """
    # Find the top k predicted classes for each sample
    num_correct_k = 0
    num_incorrect_k = 0
    top_k_preds = np.argsort(-softmax_output , axis=-1)[:, :k]

    # pdb.set_trace()
    for i in range(len(labels)):
        if int(labels[i]) in top_k_preds[i]:
            num_correct_k += 1
        else:
            num_incorrect_k += 1
            

    # Calculate recall@k as the proportion of correctly classified samples in the top k predicted classes
    recall_at_k = num_correct_k / (num_correct_k + num_incorrect_k)
    return recall_at_k


def test(eeg_model, image_model, test_dataloader):

    running_loss      = []
    eeg_featvec       = np.array([])
    image_vec         = np.array([])
    eeg_featvec_proj  = np.array([])
    image_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(test_dataloader)
    for batch_idx, (eeg, images, labels, c) in enumerate(tq, start=1):
        eeg, labels, images = eeg.to(config.device), labels.to(config.device), images.to(config.device)
        c      = c.to(config.device)
        c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
        c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
        images = torch.concat([images, c], dim=1)

        with torch.no_grad():
            x_proj = eeg_model(eeg)
            i_proj = image_model(images)
            # loss   = torch.mean(torch.square(torch.subtract(i_proj, x_proj)))
        # tq.set_description('Test:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

        eeg_featvec_proj   = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        image_featvec_proj = np.concatenate((image_featvec_proj, i_proj.cpu().detach().numpy()), axis=0) if image_featvec_proj.size else i_proj.cpu().detach().numpy()
        labels_array       = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    k_means        = K_means(n_clusters=config.num_classes)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("Test EEG KMeans score Proj: {}]".format(clustering_acc_proj))

    k_means        = K_means(n_clusters=config.num_classes)
    clustering_acc_proj = k_means.transform(image_featvec_proj, labels_array)
    print("Test Image KMeans score Proj: {}]".format(clustering_acc_proj))

    print(eeg_featvec_proj.shape, image_featvec_proj.shape, labels_array.shape)

    softmax_output  = eeg_featvec_proj @ image_featvec_proj.T
    softmax_output  = softmax(softmax_output, axis=1)

    print(softmax_output.min(axis=-1), softmax_output.max(axis=-1), softmax_output.sum(axis=-1))

    print('Recall @1: {}'.format(calculate_recall_at_k(softmax_output, labels_array, k=1)))
    print('Recall @5: {}'.format(calculate_recall_at_k(softmax_output, labels_array, k=5)))
    print('Recall @10: {}'.format(calculate_recall_at_k(softmax_output, labels_array, k=10)))


    
if __name__ == '__main__':

    base_path = config.base_path
    test_path = config.test_path
    device    = config.device

    eeg_model     = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
    eeg_model     = torch.nn.DataParallel(eeg_model).to(config.device)
   
    ckpt_path = 'eegckpt/eegfeat_all_0.9665178571428571.pth'

    checkpoint = torch.load(ckpt_path, map_location=device)
    eeg_model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading EEG checkpoint from previous epoch: {}'.format(checkpoint['epoch']))

    image_model    = ImageFeatNet(projection_dim=config.projection_dim).to(config.device)
    image_model    = torch.nn.DataParallel(image_model).to(config.device)
    experiment_num = 1
    ckpt_path      = 'EXPERIMENT_{}/bestckpt/eegfeat_all_0.6875.pth'.format(experiment_num)
    checkpoint     = torch.load(ckpt_path, map_location=device)
    image_model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading Image checkpoint from previous epoch: {}'.format(checkpoint['epoch']))

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0
    train_cluster = 0
    test_cluster   = 0

    ## Validation data
    x_test_eeg = []
    x_test_image = []
    label_test = []
    x_test_subject = []

    for i in tqdm(natsorted(os.listdir(base_path + test_path))):
        loaded_array = np.load(base_path + test_path + i, allow_pickle=True)
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
        x_test_subject.append(loaded_array[4]) # Subject Number

    x_test_eeg   = np.array(x_test_eeg)
    x_test_image = np.array(x_test_image)
    test_labels  = np.array(label_test)
    x_test_subject = np.array(x_test_subject)

    x_test_eeg   = torch.from_numpy(x_test_eeg).float()#.to(device)
    x_test_image = torch.from_numpy(x_test_image).float()#.to(device)
    test_labels  = torch.from_numpy(test_labels).long()#.to(device)
    x_test_subject = torch.from_numpy(x_test_subject).long()#.to(device)#.to(device)

    test_data       = EEGDataset(x_test_eeg, x_test_image, test_labels, x_test_subject)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)

    test(eeg_model, image_model, test_dataloader)