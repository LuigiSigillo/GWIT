import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import ViT_H_14_Weights, vit_h_14
import numpy as np
# from clip import clip
from torchmetrics.functional import accuracy
import os
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor, Normalize
from skimage import io, color

def ssim_metric(img1, img2):
    img1=np.array(img1.squeeze(0).cpu())
    img2 = np.array(img2.squeeze(0).cpu())
    img1 = np.transpose(img1, (1, 2, 0))
    img2=np.transpose(img2, (1, 2, 0))

    return ssim(img1, img2,data_range=255,channel_axis=-1)

def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device),task="multiclass",num_classes=50,
                    top_k=top_k)
        acc_list.append(acc.item())

    # print(np.mean(acc_list))
    return np.mean(acc_list), np.std(acc_list)

def preprocess_images(images):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_images = torch.stack([ToTensor()(img) for img in images])
    normalized_images = transform(tensor_images)
    return normalized_images

weights = ViT_H_14_Weights.DEFAULT
model = vit_h_14(weights=weights)
preprocess = weights.transforms()
model = model.to("cuda")
model = model.eval()
n_way = 50
num_trials = 50
top_k = 1

acc_list = []
root = "/mnt/media/luigi/model_out_CVPR_MULTISUB_CLASSIFIER_CAPTION/checkpoint-4000/controlnet/"
gt_folder = root+"ground_truth/"
gene_folder= root+"generated/"
gt_images_name = os.listdir(gt_folder)
gt_images_name.sort()
gt_image_num=0
gn_imges_name = os.listdir(gene_folder)
gn_imges_name.sort()
limit = 1
from tqdm import tqdm

for j,gt_name in tqdm(enumerate(gt_images_name)):
    # print(gt_folder + gt_name)

    # Load GT image and the path of genetrated images
    real_image = Image.open(gt_folder + gt_name).convert('RGB')

    # gene_image_name=[]
    # name1 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
    #         gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_1.png'
    # name2 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
    #         gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_2.png'
    # name3 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
    #         gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_3.png'
    # name4 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
    #         gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_4.png'
    # gene_image_name.append(name1)
    # gene_image_name.append(name2)
    # gene_image_name.append(name3)
    # gene_image_name.append(name4)

    gt = preprocess(real_image).unsqueeze(0).to("cuda")
    gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()

    generated_image_list=[]

    # Evaluate
    for i in range(0,limit):
        # print(gene_folder + gn_imges_name[i])
        generated_image = Image.open(gene_folder + gn_imges_name[j]).convert('RGB')
        pred = preprocess(generated_image).unsqueeze(0).to("cuda")
        pred_out = model(pred).squeeze(0).softmax(0).detach()
        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        generated_image_list.append(generated_image)
        acc_list.append(acc)

    gt_image_num=gt_image_num+1

print("   acc_mean:"+str(np.mean(acc_list)))


import os
import shutil
from pytorch_fid import fid_score

temp_path1 = root+'temppath'

os.makedirs(temp_path1, exist_ok=True)

for filename in os.listdir(gt_folder):
    if 'gt' not in filename:
        shutil.copy(os.path.join(gt_folder, filename), os.path.join(temp_path1, filename))

fid_value = fid_score.calculate_fid_given_paths([temp_path1, gene_folder], batch_size=50, device='cuda', dims=2048)

print('FID:', fid_value)

shutil.rmtree(temp_path1)
