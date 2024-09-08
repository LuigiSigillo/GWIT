## Take input of EEG and save it as a numpy array
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
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


if __name__ == '__main__':
    base_path = config.base_path
    train_path = config.train_path
    test_path = config.test_path
    device    = config.device

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0
    train_cluster = 0
    test_cluster   = 0

    x_train_eeg = []
    x_train_image = []
    label_train = []
    x_train_subject = []

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        label_train.append(loaded_array[2])
        x_train_subject.append(loaded_array[4]) # Subject Number
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(label_train)
    x_train_subject = np.array(x_train_subject)

    x_train_eeg   = torch.from_numpy(x_train_eeg).float()#.to(device)
    x_train_image = torch.from_numpy(x_train_image).float()#.to(device)
    train_labels  = torch.from_numpy(train_labels).long()#.to(device)
    x_train_subject = torch.from_numpy(x_train_subject).long()#.to(device)#.to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels, x_train_subject)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)



    ## Test data
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

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = ImageFeatNet(projection_dim=config.projection_dim).to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    optimizer = torch.optim.Adam(\
                                    list(model.parameters()),\
                                    lr=config.lr,\
                                    betas=(0.9, 0.999)
                                )

    
    # dir_info  = natsorted(glob('EXPERIMENT_*'))
    # if len(dir_info)==0:
    #     experiment_num = 1
    # else:
    #     experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    # if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
    #     os.makedirs('EXPERIMENT_{}'.format(experiment_num))
    #     os.makedirs('EXPERIMENT_{}/train/'.format(experiment_num))
    #     os.makedirs('EXPERIMENT_{}/val/'.format(experiment_num))
    #     os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
    #     os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
    #     os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))
    experiment_num = 1

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/bestckpt/eegfeat_all_0.6875.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:

        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))

        X_eeg_train   = np.array([])
        X_image_train = np.array([])
        Y_train       = np.array([])

        X_eeg_test   = np.array([])
        X_image_test = np.array([])
        Y_test       = np.array([])

        with torch.no_grad():
            for batch_idx, (eeg, images, labels, c) in enumerate(tqdm(train_dataloader)):
                eeg, images, labels = eeg.to(config.device), images.to(config.device), labels.to(config.device)
                c      = c.to(config.device)
                c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
                c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
                images = torch.concat([images, c], dim=1)
                i_proj   = model(images)
                X_eeg_train = np.concatenate((X_eeg_train, i_proj.cpu().detach().numpy()), axis=0) if X_eeg_train.size else i_proj.cpu().detach().numpy()
                Y_train     = np.concatenate((Y_train, labels.cpu().numpy()), axis=0) if Y_train.size else labels.cpu().numpy()
            
            for batch_idx, (eeg, images, labels, c) in enumerate(tqdm(test_dataloader)):
                eeg, images, labels = eeg.to(config.device), images.to(config.device), labels.to(config.device)
                c      = c.to(config.device)
                c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
                c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
                images = torch.concat([images, c], dim=1)
                i_proj   = model(images)
                X_eeg_test   = np.concatenate((X_eeg_test, i_proj.cpu().detach().numpy()), axis=0) if X_eeg_test.size else i_proj.cpu().detach().numpy()
                Y_test       = np.concatenate((Y_test, labels.cpu().numpy()), axis=0) if Y_test.size else labels.cpu().numpy()

    print(X_eeg_train.shape, X_image_train.shape, Y_train.shape)
    print(X_eeg_test.shape, X_image_test.shape, Y_test.shape)

    # KMeans Clustering
    num_clusters = 40
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(X_eeg_test, Y_test)
    print("[Epoch: {}, Test KMeans score Proj: {}]".format(START_EPOCH, clustering_acc_proj))

    # Source: https://www.baeldung.com/cs/svm-multiclass-classification

    poly_lin = svm.LinearSVC(multi_class='ovr', verbose=1, random_state=45, C=1.0, max_iter=1000).fit(X_eeg_train, Y_train)
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1, verbose=1, random_state=45, max_iter=1000).fit(X_eeg_train, Y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1, verbose=1, random_state=45, max_iter=1000).fit(X_eeg_train, Y_train)


    poly_lin_pred = poly_lin.predict(X_eeg_test)
    poly_pred = poly.predict(X_eeg_test)
    rbf_pred = rbf.predict(X_eeg_test)

    poly_lin_accuracy = accuracy_score(Y_test, poly_lin_pred)
    poly_lin_f1 = f1_score(Y_test, poly_lin_pred, average='weighted')
    print('Accuracy (Linear): ', "%.2f" % (poly_lin_accuracy*100))
    print('F1 (Linear): ', "%.2f" % (poly_lin_f1*100))

    poly_accuracy = accuracy_score(Y_test, poly_pred)
    poly_f1 = f1_score(Y_test, poly_pred, average='weighted')
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

    rbf_accuracy = accuracy_score(Y_test, rbf_pred)
    rbf_f1 = f1_score(Y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))