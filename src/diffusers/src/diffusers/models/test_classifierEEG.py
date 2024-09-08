from datasets import load_dataset
import sys
# sys.path.append("/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEGClip/")
# from CLIPModel import CLIPModel
# from EEG_encoder import EEG_Encoder
import torch
from torchvision.models import resnet50
from torch import nn, optim
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# sys.path.append("/home/luigi/Documents/DrEEam/src/diffusers/src/dataset_EEG/")
# from name_map_ID import name_map, folder_label_map
# # Create a reverse mapping for folder_label_map
# reverse_name_map = {v: k for k, v in name_map.items()}

# # Function to get the key from name_map using the key from folder_label_map
# def get_name_map_key(folder_label_key):
#     # Get the value from folder_label_map using the provided key
#     value = folder_label_map.get(folder_label_key)
    
#     if value is None:
#         return None  # Key not found in folder_label_map
    
#     # Get the corresponding key in name_map using the reverse mapping
#     name_map_key = reverse_name_map.get(value)
    
#     return name_map_key

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         self.label_encoder = LabelEncoder()
#         self.labels = [get_name_map_key(item['label_folder']) for item in dataset]
#         self.label_encoder.fit(self.labels)
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         eeg = item['conditioning_image']
#         image = item['image']
#         folder_label_key = item['label_folder']

#         if self.transform:
#             image = self.transform(image)
#         # Transform the string label to a numerical label
#         label = self.label_encoder.transform([get_name_map_key(folder_label_key)])[0]

#         # label = get_name_map_key(folder_label_key)

#         return eeg, image, label

# # Define the transformation to resize images
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor()
# ])




# def old_model():
#     i = 2048
#     embedding_dim  = 256
#     projection_dim = 256
#     num_layers = 1
#     device = "cuda"
#     model_path = "/home/luigi/Documents/DrEEam/src/diffusers/src/diffusers/models/checkpoints_EEG_CLIP/clip_"
#     model_path_c = model_path + str(i) + ".pth"
#     checkpoint = torch.load(model_path_c, map_location=device)
#     eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)

#     image_embedding = resnet50(pretrained=False).to(device)
#     num_features = image_embedding.fc.in_features

#     image_embedding.fc = nn.Sequential(
#         nn.ReLU(),
#         nn.Linear(num_features, embedding_dim, bias=False)
#     )

#     image_embedding.fc.to(device)

#     model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim).to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)

#     model = model.text_encoder

#     for param in model.parameters():
#         param.requires_grad = True

#     new_layer = nn.Sequential(
#         nn.Linear(embedding_dim, 256),
#         nn.ReLU(),
#         nn.Dropout(0.1),
#         nn.Linear(256, 40),
#         nn.Softmax(dim=1)
#     )

#     model.fc = nn.Sequential(
#         model.fc,
#         new_layer
#     )

#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     return model, criterion, optimizer
# def train(model, data_loader, optimizer, criterion, device, epoch):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(data_loader, 0):
#         inputs_eeg, inputs_image, labels = data
#         inputs_eeg, inputs_image, labels = inputs_eeg.to(device), inputs_image.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs_eeg.permute(0,2,1))
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     loss = running_loss / i+1
#     print('[%d] loss: %.3f' % (epoch + 1, loss))
#     return loss

# def evaluate(model,data_loader):
#     model.eval()
#     corrects = []
#     for i, data in enumerate(data_loader, 0):
#         inputs_eeg, inputs_image, labels = data
#         inputs_eeg, inputs_image, labels = inputs_eeg.to(device), inputs_image.to(device), labels.to(device)
       
#         outputs = model(inputs_eeg.permute(0,2,1))
#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted == labels).sum().item()
#         corrects.append(correct/len(labels))
#     val_acc = 100 * sum(corrects) / len(corrects)
#     print('Accuracy of the network %d %%' % (val_acc))
#     return val_acc

    
# START_EPOCH = 0
# best_val_acc   = 0.0
# best_val_epoch = 0

# data_train = load_dataset('luigi-s/EEG_Image', split='train').with_format('torch')
# data_train = CustomDataset(data_train, transform=transform)
# data_val = load_dataset('luigi-s/EEG_Image', split='validation').with_format('torch')
# data_val = CustomDataset(data_val, transform=transform)

# data_loader = torch.utils.data.DataLoader(data_train, batch_size=28, shuffle=True, num_workers=4)
# data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=28, shuffle=False, num_workers=4)
# from tqdm import tqdm
# for epoch in tqdm(range(START_EPOCH, 2049)):

#     running_train_loss = train(model, data_loader, optimizer, criterion, device, epoch)
#     val_acc   = evaluate(model, data_loader_val)


#     if best_val_acc < val_acc:
#         best_val_acc   = val_acc
#         best_val_epoch = epoch
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             }, 'eegfeat_{}_{}.pth'.format(best_val_epoch, val_acc))
# del model
# torch.cuda.empty_cache()

from pytorch_metric_learning import miners, losses
sys.path.append("/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40")
from network import EEGFeatNet
from tqdm import tqdm 
import numpy as np
from visualizations import K_means
from dataloader import EEGDataset

feat_dim       = 128 # This will give 240 dim feature
projection_dim = 128
num_layers    = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = EEGFeatNet(n_features=feat_dim, projection_dim=projection_dim, num_layers=num_layers).to(device)
model     = torch.nn.DataParallel(model).to(device)
miner   = miners.MultiSimilarityMiner()
model.load_state_dict(torch.load("/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth")['model_state_dict'])

def test(epoch, model, knn_cv, loss_fn, miner, test_dataloader, experiment_num):
    running_loss      = []
    # eeg_featvec       = np.array([])
    image_vec         = np.array([])
    eeg_featvec_proj  = np.array([])
    # eeg_gamma         = np.array([])
    labels_array      = np.array([])
    tq = tqdm(test_dataloader)
    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        corrects = []
        eeg, labels = eeg.to(device), labels.to(device)

        with torch.no_grad():
            x_proj = model(eeg)
            hard_pairs = miner(x_proj, labels)
            loss       = loss_fn(x_proj, labels, hard_pairs)
            running_loss = running_loss + [loss.detach().cpu().numpy()]
            # Predict the labels
            predicted_labels = knn_cv.predict(x_proj.cpu().detach().numpy())
            correct = (predicted_labels == labels.cpu().detach().numpy()).sum().item()
            corrects.append(correct/len(labels))
        tq.set_description('Test:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))
        image_vec        = np.concatenate((image_vec, images.cpu().detach().numpy()), axis=0) if image_vec.size else images.cpu().detach().numpy()
	    # eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    val_acc = 100 * sum(corrects) / len(corrects)
    print('Accuracy of the network %d %%' % (val_acc))
    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = 40 #config.num_classes
	# k_means        = K_means(n_clusters=num_clusters)
	# clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
	# print("[Epoch: {}, Val KMeans score Feat: {}]".format(epoch, clustering_acc_feat))
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Test KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# reduced_embedding = tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj')

	# umap_plot = Umap()
	# reduced_embedding = umap_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj')
	
	# visualize_scatter_with_images(reduced_embedding, images = [np.uint8(cv2.resize(np.transpose(i, (1, 2, 0)), (45,45))) for i in image_vec], image_zoom=0.7)

	# tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	# reduced_embedding = tsne_plot.plot3d(eeg_featvec_proj, labels_array, clustering_acc_proj, 'test', experiment_num, epoch, proj_type='proj')
	
	# visualize_scatter_with_images3d(reduced_embedding, images = [np.uint8(cv2.resize(np.transpose(i, (1, 2, 0)), (45,45))) for i in image_vec], labels=labels_array, image_zoom=0.7)

	#create a new KNN model
    knn_cv = KNeighborsClassifier(n_neighbors=3)
    # cv_scores = cross_val_score(knn_cv, eeg_featvec_proj, labels_array, cv=5)#print each cv score (accuracy) and average them
    # print(cv_scores)
    # print('cv_scores mean:{}'.format(np.mean(cv_scores)))
    knn_cv.fit(eeg_featvec_proj, labels_array)

    # Predict the labels
    predicted_labels = knn_cv.predict(eeg_featvec_proj)

    # Print the predicted labels
    print(predicted_labels)
    return running_loss, clustering_acc_proj

loss_fn = losses.TripletMarginLoss()
epoch = 0
lr = 3e-4
model.eval()
batch_size = 28


## Validation data
x_test_eeg = []
x_test_image = []
label_test = []
x_train_eeg = []
x_train_image = []
label_train = []

base_path       = '/mnt/media/luigi/dataset/dreamdiff/'
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'eeg_imagenet40_cvpr_2017_raw/test/'
import cv2
from natsort import natsorted
import os
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

x_test_eeg   = np.array(x_test_eeg)
x_test_image = np.array(x_test_image)
test_labels  = np.array(label_test)
x_test_eeg   = torch.from_numpy(x_test_eeg).float().to(device)
x_test_image = torch.from_numpy(x_test_image).float().to(device)
test_labels  = torch.from_numpy(test_labels).long().to(device)
test_data       = EEGDataset(x_test_eeg, x_test_image, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)

#train data to fit knn
for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    x_train_eeg.append(loaded_array[1].T)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    x_train_image.append(img)
    # if loaded_array[3] not in class_labels:
    # 	class_labels[loaded_array[3]] = label_count
    # 	label_count += 1
    # 	test_cluster += 1
    # label_test.append(class_labels[loaded_array[3]])
    label_train.append(loaded_array[2])
x_train_eeg   = np.array(x_train_eeg)
x_train_image = np.array(x_train_image)
train_labels  = np.array(label_train)
x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(device)
x_train_image = torch.from_numpy(x_train_image).float().to(device)
train_labels  = torch.from_numpy(train_labels).long().to(device)
train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

x_proj_train = torch.stack([model(batch[0].float().to(device)) for batch in train_dataloader])
train_labels_batch = torch.stack([batch[2]for batch in train_dataloader])
knn_cv = KNeighborsClassifier(n_neighbors=3)

knn_cv.fit(x_proj_train.view(-1,128).cpu().detach().numpy(), train_labels_batch.view(-1).cpu().detach().numpy())
# Save the model to a file
import pickle

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_cv, f)

# running_test_loss, test_acc   = test(epoch, model,knn_cv, loss_fn, miner, test_dataloader, 29)



