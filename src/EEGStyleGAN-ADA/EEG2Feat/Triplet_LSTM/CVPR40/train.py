## Take input of EEG and save it as a numpy array
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config
from tqdm import tqdm
import numpy as np
import pdb
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

np.random.seed(45)
torch.manual_seed(45)

def train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num):
        
    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(train_dataloader)
    # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tq):
    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)
        eeg    = eeg.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()

        # x1_proj, x1 = model(eeg_x1)
        # x2_proj, x2 = model(eeg_x2)
        x_proj = model(eeg)

        hard_pairs = miner(x_proj, labels)
        loss       = loss_fn(x_proj, labels, hard_pairs)
        
        # loss  = loss_fn(x1_proj, x2_proj)
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%config.vis_freq) == 0:
        # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
        for batch_idx, (eeg, images, labels) in enumerate(tqdm(train_dataloader)):
            eeg, labels = eeg.to(config.device), labels.to(config.device)
            with torch.no_grad():
                x_proj = model(eeg)
            # eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
            eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
            # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
            labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

        ### compute k-means score and Umap score on the text and image embeddings
        num_clusters   = 40
        # k_means        = K_means(n_clusters=num_clusters)
        # clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
        # print("[Epoch: {}, Train KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

        k_means        = K_means(n_clusters=num_clusters)
        clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
        print("[Epoch: {}, Train KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

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

	tq = tqdm(val_dataloader)
	for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
		eeg, labels = eeg.to(config.device), labels.to(config.device)
		with torch.no_grad():
			x_proj = model(eeg)

			hard_pairs = miner(x_proj, labels)
			loss       = loss_fn(x_proj, labels, hard_pairs)

			running_loss = running_loss + [loss.detach().cpu().numpy()]

		tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

		# eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
		eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
		# eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
		labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

	### compute k-means score and Umap score on the text and image embeddings
	num_clusters   = 40
	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
	# print("[Epoch: {}, Val KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

	k_means        = K_means(n_clusters=num_clusters)
	clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
	print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_gamma = k_means.transform(eeg_gamma, labels_array)
	# print("[Epoch: {}, KMeans score gamma: {}]".format(epoch, clustering_acc_proj))

	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'val', experiment_num, epoch, proj_type='feat')


	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'val', experiment_num, epoch, proj_type='proj')

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

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # labels.append(class_labels[loaded_array[3]])
        labels.append(loaded_array[2])
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)

    # ## convert numpy array to tensor
    x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(device)
    x_train_image = torch.from_numpy(x_train_image).float().to(device)
    train_labels  = torch.from_numpy(train_labels).long().to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


    ## Validation data
    x_val_eeg = []
    x_val_image = []
    label_Val = []

    for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
        x_val_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_val_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # label_Val.append(class_labels[loaded_array[3]])
        label_Val.append(loaded_array[2])
        
    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    val_labels  = np.array(label_Val)

    x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
    x_val_image = torch.from_numpy(x_val_image).float().to(device)
    val_labels  = torch.from_numpy(val_labels).long().to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    optimizer = torch.optim.Adam(\
                                    list(model.parameters()),\
                                    lr=config.lr,\
                                    betas=(0.9, 0.999)
                                )

    
    dir_info  = natsorted(glob('EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_ckpt/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_bestckpt/'.format(experiment_num))
        os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    miner   = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()
    # loss_fn = ContrastiveLoss(batch_size=config.batch_size, temperature=config.temperature)
    # loss_fn = PerceptualLoss()
    # loss_fn   = F.l1_loss
    # loss_fn = lpips.LPIPS(net='vgg').to(config.device)
    # loss_fn  = nn.MSELoss()
    # loss_fn  = nn.CrossEntropyLoss()
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
              }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))


        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))

        # running_val_loss   = validation(epoch, model, optimizer, loss_fn, train_data, val_dataloader)
        # print(np.mean(running_train_loss), eeg_featvec.shape, labels_array.shape)

        # if (epoch%1) == 0:
        #     ### compute k-means score and Umap score on the text and image embeddings
        #     num_clusters = 40
        #     k_means        = K_means(n_clusters=num_clusters)
        #     clustering_acc = k_means.transform(eeg_featvec, labels_array)
        #     print("KMeans score:", clustering_acc)

        #     with torch.no_grad():
        #         pred = model(base_spectrograms)[0]
        #         gt   = base_spectrograms[0]

        #     save_image(pred, gt, experiment_num, epoch, 'val')
        # break
        # validate(model, 0.1, train_data)

        # print('completed')
