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
from torchvision.models import ResNet18_Weights


np.random.seed(45)
torch.manual_seed(45)

image_preprocessing = ResNet18_Weights.IMAGENET1K_V1.transforms()

def train(epoch, model, image_model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num):
        
    running_loss      = []
    image_featvec       = np.array([])
    image_featvec_proj  = np.array([])
    image_gamma         = np.array([])
    labels_array        = np.array([])
    image_pred          = np.array([])

    tq = tqdm(train_dataloader)
    # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tq):
    for batch_idx, (eeg, images, labels, c) in enumerate(tq, start=1):
        # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)
        eeg    = eeg.to(config.device)
        labels = labels.to(config.device)
        images = image_preprocessing(images.to(config.device))
        c      = c.to(config.device)
        c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
        c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
        images = torch.concat([images, c], dim=1)
        optimizer.zero_grad()
        # x1_proj, x1 = model(eeg_x1)
        # x2_proj, x2 = model(eeg_x2)
        x_proj = model(eeg)
        i_proj = image_model(images)

        # hard_pairs = miner(i_proj, labels)
        # loss       = loss_fn(i_proj, labels, hard_pairs) + torch.mean(torch.square(torch.subtract(x_proj, i_proj)))
        loss       = torch.mean(torch.square(torch.subtract(x_proj, i_proj)))
        
        # loss  = loss_fn(x1_proj, x2_proj)
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%config.vis_freq) == 0:
        # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
        for batch_idx, (eeg, images, labels, c) in enumerate(tqdm(train_dataloader)):
            eeg, labels, images = eeg.to(config.device), labels.to(config.device), image_preprocessing(images.to(config.device))
            c      = c.to(config.device)
            c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
            c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
            images = torch.concat([images, c], dim=1)
            with torch.no_grad():
                i_proj = image_model(images)

            image_featvec_proj = np.concatenate((image_featvec_proj, i_proj.cpu().detach().numpy()), axis=0) if image_featvec_proj.size else i_proj.cpu().detach().numpy()
            # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
            labels_array = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

        ### compute k-means score and Umap score on the text and image embeddings
        num_clusters   = config.num_classes
        # k_means        = K_means(n_clusters=num_clusters)
        # clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)

        k_means        = K_means(n_clusters=num_clusters)
        clustering_acc_proj = k_means.transform(image_featvec_proj, labels_array)
        print("[Epoch: {}, Train KMeans score Feat: {}]".format(epoch, clustering_acc_proj))

        # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        # tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'train', experiment_num, epoch, proj_type='feat')
        

        # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        # tsne_plot.plot(image_featvec_proj, labels_array, clustering_acc_proj, 'train', experiment_num, epoch, proj_type='proj')

    return running_loss
 

def validation(epoch, model, image_model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num):

    running_loss      = []
    images_featvec       = np.array([])
    image_featvec_proj  = np.array([])
    images_gamma         = np.array([])
    labels_array      = np.array([])
    images_pred          = np.array([])
    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, images, labels, c) in enumerate(tq, start=1):
        eeg, labels, images = eeg.to(config.device), labels.to(config.device), image_preprocessing(images.to(config.device))
        c      = c.to(config.device)
        c 	   = torch.unsqueeze( torch.unsqueeze(c, dim=-1), dim=-1 )
        c 	   = torch.tile(c, (1, 1, images.shape[2], images.shape[3]))
        images = torch.concat([images, c], dim=1)
        with torch.no_grad():
            x_proj = model(eeg)
            i_proj = image_model(images)
            # hard_pairs = miner(i_proj, labels)
            # loss       = loss_fn(i_proj, labels, hard_pairs) + torch.mean(torch.square(torch.subtract(x_proj, i_proj)))
            loss   = torch.mean(torch.square(torch.subtract(x_proj, i_proj)))

        running_loss = running_loss + [loss.detach().cpu().numpy()]
        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))
        image_featvec_proj = np.concatenate((image_featvec_proj, i_proj.cpu().detach().numpy()), axis=0) if image_featvec_proj.size else i_proj.cpu().detach().numpy()
		# eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = config.num_classes
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(image_featvec_proj, labels_array)
    print("[Epoch: {}, Val KMeans score Feat: {}]".format(epoch, clustering_acc_proj))

    return running_loss, clustering_acc_proj

    
if __name__ == '__main__':

    base_path       = config.base_path
    train_path      = config.train_path
    validation_path = config.validation_path
    device          = config.device

            
    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []
    x_train_subject = []

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(np.expand_dims(loaded_array[1].T, axis=0))
        # img = cv2.resize(loaded_array[0], (config.image_shape[1], config.image_shape[2]))
        # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = cv2.resize(loaded_array[0], (224, 224))
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # labels.append(class_labels[loaded_array[3]])
        labels.append(loaded_array[2])
        x_train_subject.append(loaded_array[4]) # Subject Number
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)
    # x_train_subject = np.expand_dims(np.expand_dims(np.expand_dims(np.array(x_train_subject), axis=-1), axis=-1), axis=-1)
    x_train_subject = np.array(x_train_subject)

    # ## convert numpy array to tensor
    x_train_eeg   = torch.from_numpy(x_train_eeg).float()#.to(device)
    x_train_image = torch.from_numpy(x_train_image).float()#.to(device)
    train_labels  = torch.from_numpy(train_labels).long()#.to(device)
    x_train_subject = torch.from_numpy(x_train_subject).long()#.to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels, x_train_subject)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


    ## Validation data
    x_val_eeg = []
    x_val_image = []
    label_Val = []
    x_val_subject = []

    for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
        x_val_eeg.append(np.expand_dims(loaded_array[1].T, axis=0))
        # img = cv2.resize(loaded_array[0], (config.image_shape[1], config.image_shape[2]))
        # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = cv2.resize(loaded_array[0], (224, 224))
        img = np.transpose(img, (2, 0, 1))
        x_val_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # label_Val.append(class_labels[loaded_array[3]])
        label_Val.append(loaded_array[2])
        x_val_subject.append(loaded_array[4])
        
    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    val_labels  = np.array(label_Val)
    # x_val_subject = np.expand_dims(np.expand_dims(np.expand_dims(np.array(x_val_subject), axis=-1), axis=-1), axis=-1)
    x_val_subject = np.array(x_val_subject)


    x_val_eeg   = torch.from_numpy(x_val_eeg).float()#.to(device)
    x_val_image = torch.from_numpy(x_val_image).float()#.to(device)
    val_labels  = torch.from_numpy(val_labels).long()#.to(device)
    x_val_subject = torch.from_numpy(x_val_subject).long()#.to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image, val_labels, x_val_subject)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = EEGFeatNet(input_shape=config.input_shape, n_features=config.feat_dim, projection_dim=config.projection_dim).to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    
    experiment_num = 2
    
    ckpt_lst = natsorted(glob('EXPERIMENT_{}/bestckpt/eegfeat_all_0.9771205357142857.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Loading checkpoint from previous epoch: {}'.format(checkpoint['epoch']))

    for params in model.module.parameters():
        params.requires_grad = False

    image_model = ImageFeatNet(projection_dim=config.projection_dim).to(config.device)

    optimizer = torch.optim.Adam(\
                                    list(image_model.parameters()),\
                                    lr=config.lr,\
                                    betas=(0.9, 0.999)
                                )

    # miner = None
    miner   = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()
    # loss_fn = ContrastiveLoss(batch_size=config.batch_size, temperature=config.temperature)
    # loss_fn = PerceptualLoss()
    # loss_fn   = F.l1_loss
    # loss_fn = lpips.LPIPS(net='vgg').to(config.device)
    # loss_fn  = nn.MSELoss()
    # loss_fn  = nn.PairwiseDistance(p=2.0, eps=1e-06)
    # base_eeg, base_images, base_labels, base_spectrograms = next(iter(val_dataloader))
    # base_eeg, base_images = base_eeg.to(config.device), base_images.to(config.device)
    # base_labels, base_spectrograms = base_labels.to(config.device), base_spectrograms.to(config.device)
    best_val_acc   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, EPOCHS):

        running_train_loss = train(epoch, model, image_model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num)
        if (epoch%config.vis_freq) == 0:
        	running_val_loss, val_acc   = validation(epoch, model, image_model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num)

        if best_val_acc < val_acc:
        	best_val_acc   = val_acc
        	best_val_epoch = epoch
        	torch.save({
                'epoch': epoch,
                'image_model_state_dict': image_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/imagemodel_bestckpt/imagefeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))


        torch.save({
                'epoch': epoch,
                'image_model_state_dict': image_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/imagemodel_ckpt/imagefeat_{}.pth'.format(experiment_num, 'all'))
