## Take input of EEG and save it as a numpy array
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from network import EEGFeatNet
# from model import ModifiedResNet
# from CLIPModel import CLIPModel
from visualizations import Umap, K_means, TsnePlot, save_image
from losses import ContrastiveLoss
from dataaugmentation import apply_augmentation
import pickle

np.random.seed(45)
torch.manual_seed(45)

def train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num):
        
    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])
    eeg_pred          = np.array([])

    tq = tqdm(train_dataloader)
    # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tq):
    for batch_idx, (eeg, images, labels, _) in enumerate(tq, start=1):
        # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)
        eeg    = eeg.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()
        # x1_proj, x1 = model(eeg_x1)
        # x2_proj, x2 = model(eeg_x2)
        y_cap = model(eeg)

        # hard_pairs = miner(x_proj, labels)
        loss  = loss_fn(y_cap, labels)
        
        # loss  = loss_fn(x1_proj, x2_proj)
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%config.vis_freq) == 0:
        # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
        for batch_idx, (eeg, images, labels, _) in enumerate(tqdm(train_dataloader)):
            eeg, labels = eeg.to(config.device), labels.to(config.device).to(torch.float32)
            with torch.no_grad():
                _, y_cap = torch.max(model(eeg), 1)
            eeg_pred     = np.concatenate((eeg_pred, y_cap.cpu().detach().numpy()), axis=0) if eeg_pred.size else y_cap.cpu().detach().numpy()
            # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
            labels_array = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

        ### compute k-means score and Umap score on the text and image embeddings
        # num_clusters   = 40
        # k_means        = K_means(n_clusters=num_clusters)
        # clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
        # print("[Epoch: {}, Train KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

        # k_means        = K_means(n_clusters=num_clusters)
        # clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
        pred_acc = np.mean(eeg_pred==labels_array)
        print("[Epoch: {}, Train Acc: {}]".format(epoch, pred_acc))

        # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        # tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'train', experiment_num, epoch, proj_type='feat')
        

        # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        # tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'train', experiment_num, epoch, proj_type='proj')

    return running_loss
 

def validation(epoch, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num):

    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])
    eeg_pred          = np.array([])
    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, images, labels, _) in enumerate(tq, start=1):
        eeg, labels = eeg.to(config.device), labels.to(config.device)
        with torch.no_grad():
            y_cap = model(eeg)
            loss  = loss_fn(y_cap, labels)
            _, y_cap = torch.max(y_cap, 1)

        running_loss = running_loss + [loss.detach().cpu().numpy()]
        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))
        eeg_pred      = np.concatenate((eeg_pred, y_cap.cpu().detach().numpy()), axis=0) if eeg_pred.size else y_cap.cpu().detach().numpy()
		# eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
		# eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

	### compute k-means score and Umap score on the text and image embeddings
	# num_clusters   = 40
	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
	# print("[Epoch: {}, Val KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    pred_acc = np.mean(eeg_pred==labels_array)
    print("[Epoch: {}, Val Acc: {}]".format(epoch, pred_acc))

	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_gamma = k_means.transform(eeg_gamma, labels_array)
	# print("[Epoch: {}, KMeans score gamma: {}]".format(epoch, clustering_acc_proj))

	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'val', experiment_num, epoch, proj_type='feat')


	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'val', experiment_num, epoch, proj_type='proj')
    return running_loss, pred_acc

    
if __name__ == '__main__':

    base_path       = config.base_path
    train_path      = config.train_path
    validation_path = config.validation_path
    device          = config.device


    with open(base_path + config.thoughtviz_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        val_X = data['x_test']
        val_Y = data['y_test']

    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []
    x_train_subject=[]

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = 1000#config.epoch

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


    ## Validation data
    x_val_eeg   = []
    x_val_image = []
    label_val   = []
    x_val_subject = []

    for idx in tqdm(range(val_X.shape[0])):
        # x_val_eeg.append(np.transpose(val_X[idx], (2, 1, 0)))
        x_val_eeg.append(np.squeeze(np.transpose(val_X[idx], (2, 1, 0)), axis=0))
        x_val_image.append(np.zeros(shape=(2, 2)))
        x_val_subject.append(0.0)
        label_val.append(np.argmax(val_Y[idx]))

    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    label_val   = np.array(label_val)
    x_val_subject = np.array(x_val_subject)

    print(x_val_eeg.shape, x_val_image.shape, label_val.shape, x_val_subject.shape)
    print('Total number of classes: {}'.format(len(np.unique(label_val))), np.unique(label_val))

    x_val_eeg   = torch.from_numpy(x_val_eeg).float()#.to(device)
    x_val_image = torch.from_numpy(x_val_image).float()#.to(device)
    label_val   = torch.from_numpy(label_val).long()#.to(device)
    x_val_subject  = torch.from_numpy(x_val_subject).long()#.to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image, label_val, x_val_subject)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = EEGFeatNet(n_classes=config.num_classes, in_channels=config.input_size,\
                           n_features=config.feat_dim, projection_dim=config.projection_dim,\
                           num_layers=config.num_layers).to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    
    experiment_num = 1
    
    ckpt_path = 'EXPERIMENT_{}/bestckpt/eegfeat_all_0.7212357954545454.pth'.format(experiment_num)

    START_EPOCH = 0

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    for params in model.module.parameters():
        params.requires_grad = True

    model.module.fc  = nn.Sequential(
                                model.module.fc,
                                nn.LeakyReLU(),
                                nn.Linear(in_features=model.module.fc.out_features, out_features=config.num_classes, bias=True),
                                # nn.Softmax(dim=1)
                            ).to(config.device)

    optimizer = torch.optim.Adam(\
                                    list(model.parameters()),\
                                    lr=config.lr,\
                                    betas=(0.9, 0.999)
                                )

    miner = None
    # miner   = miners.MultiSimilarityMiner()
    # loss_fn = losses.TripletMarginLoss()
    # loss_fn = ContrastiveLoss(batch_size=config.batch_size, temperature=config.temperature)
    # loss_fn = PerceptualLoss()
    # loss_fn   = F.l1_loss
    # loss_fn = lpips.LPIPS(net='vgg').to(config.device)
    # loss_fn  = nn.MSELoss()
    loss_fn  = nn.CrossEntropyLoss()
    # base_eeg, base_images, base_labels, base_spectrograms = next(iter(val_dataloader))
    # base_eeg, base_images = base_eeg.to(config.device), base_images.to(config.device)
    # base_labels, base_spectrograms = base_labels.to(config.device), base_spectrograms.to(config.device)
    best_val_acc   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, EPOCHS):

        running_train_loss = train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num)
        if (epoch%config.vis_freq) == 0:
        	running_val_loss, val_acc   = validation(epoch, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num)

        if best_val_acc < val_acc:
        	best_val_acc   = val_acc
        	best_val_epoch = epoch
        	torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/finetune_bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))
