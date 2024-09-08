import os
import cv2
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import math
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from EEG_encoder import EEG_Encoder
from model import ModifiedResNet
from CLIPModel import CLIPModel
from Visualizations import Umap, K_means, TsnePlot


if __name__ == '__main__':
    train = []
    test  = []
    idx_to_class = {}
    class_to_idx = {}
    idx = 0
    data_path = '../dataset/'
    dim = 512

    ## Training data
    x_train_eeg   = []
    x_train_image = []
    labels_train  = []

    for i in tqdm(natsorted(os.listdir(config.base_path + config.train_path))):
        loaded_array = np.load(config.base_path + config.train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        labels_train.append(loaded_array[2])
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    labels_train  = np.array(labels_train)
    x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(config.device)
    x_train_image = torch.from_numpy(x_train_image).float().to(config.device)
    labels_train  = torch.from_numpy(labels_train).long().to(config.device)
    train_data        = torch.utils.data.TensorDataset(x_train_eeg, x_train_image, labels_train)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = config.batch_size, shuffle=True)

    ## Test data
    x_test_eeg   = []
    x_test_image = []
    label_test   = []

    for i in tqdm(natsorted(os.listdir(config.base_path + config.test_path))):
        loaded_array = np.load(config.base_path + config.test_path + i, allow_pickle=True)
        x_test_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = np.transpose(img, (2, 0, 1))
        x_test_image.append(img)
        label_test.append(loaded_array[2])
        
    x_test_eeg   = np.array(x_test_eeg)
    x_test_image = np.array(x_test_image)
    labels_test  = np.array(label_test)

    x_test_eeg   = torch.from_numpy(x_test_eeg).float().to(config.device)
    x_test_image = torch.from_numpy(x_test_image).float().to(config.device)
    labels_test  = torch.from_numpy(labels_test).long().to(config.device)

    test_data        = torch.utils.data.TensorDataset(x_test_eeg, x_test_image, labels_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = config.batch_size, shuffle=False)


    # Extracting Features
    eeg_embedding = EEG_Encoder(config.input_size, config.hidden_size, config.num_layers, device=config.device).to(config.device)
    image_embedding = ModifiedResNet(layers = [3, 4, 6, 3], output_dim=config.embedding_dim, heads=8).to(config.device)

    model = CLIPModel(eeg_embedding, image_embedding, config.embedding_dim, config.projection_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    dir_info       = natsorted(glob('EXPERIMENT_*'))
    experiment_num = 1
    ckpt_lst       = natsorted(glob('EXPERIMENT_{}/bestckpt_54_56/nerf_*.pth'.format(experiment_num)))

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
            for batch_idx, (texts, images, labels) in enumerate(tqdm(train_data_loader)):
                # get the embeddings for the text and images
                text_embed, image_embed, text_feat, image_feat = model(texts, images)
                X_eeg_train   = np.concatenate((X_eeg_train, text_embed.cpu().detach().numpy()), axis=0) if X_eeg_train.size else text_embed.cpu().detach().numpy()
                X_image_train = np.concatenate((X_image_train, image_embed.cpu().detach().numpy()), axis=0) if X_image_train.size else image_embed.cpu().detach().numpy()
                Y_train       = np.concatenate((Y_train, labels.cpu().numpy()), axis=0) if Y_train.size else labels.cpu().numpy()
            
            for batch_idx, (texts, images, labels) in enumerate(tqdm(test_data_loader)):
                # get the embeddings for the text and images
                text_embed, image_embed, text_feat, image_feat = model(texts, images)
                X_eeg_test   = np.concatenate((X_eeg_test, text_embed.cpu().detach().numpy()), axis=0) if X_eeg_test.size else text_embed.cpu().detach().numpy()
                X_image_test = np.concatenate((X_image_test, image_embed.cpu().detach().numpy()), axis=0) if X_image_test.size else image_embed.cpu().detach().numpy()
                Y_test       = np.concatenate((Y_test, labels.cpu().numpy()), axis=0) if Y_test.size else labels.cpu().numpy()

    print(X_eeg_train.shape, X_image_train.shape, Y_train.shape)
    print(X_eeg_test.shape, X_image_test.shape, Y_test.shape)

    # KMeans Clustering
    num_clusters = 40
    k_means      = K_means(n_clusters=num_clusters)
    (clustered_text_label, clustered_image_label), (text_score, image_score) = k_means.transform(X_eeg_train, X_image_train)
    print(f"Text KMeans score:", text_score)
    print(f"Image KMeans score:", image_score)

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