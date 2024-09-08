## Take input of EEG and save it as a numpy array
import config
from tqdm import tqdm
import numpy as np
import pdb
import os
from natsort import natsorted
import cv2
import pickle
from glob import glob
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses

import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import EEGDataset
from network import EEGFeatNet
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

    with open(base_path + config.thoughtviz_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        test_X = data['x_test']
        test_Y = data['y_test']

    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []
    x_train_subject=[]

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    for idx in tqdm(range(train_X.shape[0])):
        # x_train_eeg.append(np.transpose(train_X[idx], (2, 1, 0)))
        x_train_eeg.append(np.squeeze(np.transpose(train_X[idx], (2, 1, 0)), axis=0))
        x_train_image.append(np.zeros(shape=(2, 2)))
        x_train_subject.append(0)
        labels.append(np.argmax(train_Y[idx]))
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)
    x_train_subject = np.array(x_train_subject)

    print(x_train_eeg.shape, x_train_image.shape, train_labels.shape, x_train_subject.shape)
    print('Total number of classes: {}'.format(len(np.unique(train_labels))), np.unique(train_labels))

    # ## convert numpy array to tensor
    x_train_eeg   = torch.from_numpy(x_train_eeg).float()#.to(device)
    x_train_image = torch.from_numpy(x_train_image).float()#.to(device)
    train_labels  = torch.from_numpy(train_labels).long()#.to(device)
    x_train_subject  = torch.from_numpy(x_train_subject).long()#.to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels, x_train_subject)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []
    x_train_subject=[]

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    ## Validation data
    x_test_eeg   = []
    x_test_image = []
    label_test   = []
    x_test_subject = []

    for idx in tqdm(range(test_X.shape[0])):
        # x_test_eeg.append(np.transpose(test_X[idx], (2, 1, 0)))
        x_test_eeg.append(np.squeeze(np.transpose(test_X[idx], (2, 1, 0)), axis=0))
        x_test_image.append(np.zeros(shape=(2, 2)))
        x_test_subject.append(0.0)
        label_test.append(np.argmax(test_Y[idx]))

    x_test_eeg   = np.array(x_test_eeg)
    x_test_image = np.array(x_test_image)
    label_test   = np.array(label_test)
    x_test_subject = np.array(x_test_subject)

    print(x_test_eeg.shape, x_test_image.shape, label_test.shape, x_test_subject.shape)
    print('Total number of classes: {}'.format(len(np.unique(label_test))), np.unique(label_test))

    x_test_eeg   = torch.from_numpy(x_test_eeg).float().to(device)
    x_test_image = torch.from_numpy(x_test_image).float()#.to(device)
    label_test   = torch.from_numpy(label_test).long().to(device)
    x_test_subject  = torch.from_numpy(x_test_subject).long()#.to(device)

    test_data       = EEGDataset(x_test_eeg, x_test_image, label_test, x_test_subject)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = EEGFeatNet(n_classes=config.num_classes, in_channels=config.input_size,\
                           n_features=config.feat_dim, projection_dim=config.projection_dim,\
                           num_layers=config.num_layers).to(config.device)
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

    ckpt_path = 'EXPERIMENT_{}/bestckpt/eegfeat_all_0.7212357954545454.pth'.format(experiment_num)

    START_EPOCH = 0

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
        for batch_idx, (eeg, images, labels, _) in enumerate(tqdm(train_dataloader)):
            eeg, labels = eeg.to(config.device), labels.to(config.device)
            x_proj      = model(eeg)
            X_eeg_train = np.concatenate((X_eeg_train, x_proj.cpu().detach().numpy()), axis=0) if X_eeg_train.size else x_proj.cpu().detach().numpy()
            Y_train     = np.concatenate((Y_train, labels.cpu().numpy()), axis=0) if Y_train.size else labels.cpu().numpy()
        
        for batch_idx, (eeg, images, labels, _) in enumerate(tqdm(test_dataloader)):
            # get the embeddings for the text and images
            eeg, labels = eeg.to(config.device), labels.to(config.device)
            x_proj      = model(eeg)
            X_eeg_test   = np.concatenate((X_eeg_test, x_proj.cpu().detach().numpy()), axis=0) if X_eeg_test.size else x_proj.cpu().detach().numpy()
            Y_test       = np.concatenate((Y_test, labels.cpu().numpy()), axis=0) if Y_test.size else labels.cpu().numpy()

    print(X_eeg_train.shape, X_image_train.shape, Y_train.shape)
    print(X_eeg_test.shape, X_image_test.shape, Y_test.shape)

    # KMeans Clustering
    num_clusters = config.num_classes

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