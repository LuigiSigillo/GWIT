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
import pickle

np.random.seed(45)
torch.manual_seed(45)

def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)

    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()

def visualize_scatter_with_images3d(X_3d_data, images, labels, figsize=(45,45), image_zoom=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    # ax.axis([-100, 100, -100, 100])
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
    	mask = labels == label
    	ax.scatter(X_3d_data[mask, 0], X_3d_data[mask, 1], X_3d_data[mask, 2], c=colors[i], label=label)

    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111,frame_on=False) 
    ax2.axis("off")
    # ax2.axis([-100,100,-100,100])
    ia = ImageAnnotations3D(np.c_[X_3d_data[:, 0],X_3d_data[:, 1],X_3d_data[:, 2]],images,ax, ax2 )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



def test(epoch, model, optimizer, loss_fn, miner, test_dataloader, experiment_num):

    running_loss      = []
    eeg_featvec       = np.array([])
    image_vec         = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(test_dataloader)
    for batch_idx, (eeg, images, labels, _) in enumerate(tq, start=1):
        eeg, labels = eeg.to(config.device), labels.to(config.device)
        with torch.no_grad():
            x_proj     = model(eeg)
            hard_pairs = miner(x_proj, labels)
            loss       = loss_fn(x_proj, labels, hard_pairs)

            running_loss = running_loss + [loss.detach().cpu().numpy()]

            tq.set_description('Test:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = 10

    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Test KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

    tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    reduced_embedding = tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj_thoughtviz')

    umap_plot = Umap()
    reduced_embedding = umap_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj_thoughtviz')

    # visualize_scatter_with_images(reduced_embedding, images = [np.uint8(cv2.resize(np.transpose(i, (1, 2, 0)), (45,45))) for i in image_vec], image_zoom=0.7)

    # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    # reduced_embedding = tsne_plot.plot3d(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj')

    # visualize_scatter_with_images3d(reduced_embedding, images = [np.uint8(cv2.resize(np.transpose(i, (1, 2, 0)), (45,45))) for i in image_vec], labels=labels_array, image_zoom=0.7)

    return running_loss, clustering_acc_proj


if __name__ == '__main__':

    base_path = config.base_path
    test_path = config.test_path
    device    = config.device

    # ## hyperparameters
    batch_size     = config.test_batch_size
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

    ## Validation data
    x_test_eeg = []
    x_test_image = []
    label_test = []
    x_test_subject = []

    for idx in tqdm(range(test_X.shape[0])):
        # loaded_array = np.load(base_path + test_path + i, allow_pickle=True)
        # x_test_eeg.append(loaded_array[1].T)
        # print(np.transpose(test_X[idx], (2, 1, 0)).shape)
        x_test_eeg.append(np.transpose(test_X[idx], (2, 1, 0)))
        # img = cv2.resize(loaded_array[0], (224, 224))
        # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        # img = np.transpose(img, (2, 0, 1))
        x_test_image.append(np.zeros(shape=(2, 2)))
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # 	test_cluster += 1
        # label_test.append(class_labels[loaded_array[3]])
        label_test.append(np.argmax(test_Y[idx]))
        x_test_subject.append(0)

    x_test_eeg   = np.array(x_test_eeg)
    x_test_image = np.array(x_test_image)
    test_labels  = np.array(label_test)
    x_test_subject = np.array(x_test_subject)

    x_test_eeg   = torch.from_numpy(x_test_eeg).float().to(device)
    x_test_image = torch.from_numpy(x_test_image).float().to(device)
    test_labels  = torch.from_numpy(test_labels).long().to(device)
    x_test_subject  = torch.from_numpy(x_test_subject).long().to(device)

    test_data       = EEGDataset(x_test_eeg, x_test_image, test_labels, x_test_subject)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)

    # model     = CNNEEGFeatureExtractor(input_shape=[1, config.input_size, config.timestep],\
    #                                    feat_dim=config.feat_dim,\
    #                                    projection_dim=config.projection_dim).to(config.device)
    model     = EEGFeatNet(input_shape=config.input_shape, n_features=config.feat_dim, projection_dim=config.projection_dim).to(config.device)
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
    experiment_num = 2

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/bestckpt/eegfeat_all_0.9642857142857143.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    else:
        os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    miner   = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()
    epoch   = START_EPOCH

    running_test_loss, test_acc   = test(epoch, model, optimizer, loss_fn, miner, test_dataloader, experiment_num)