from torch import nn
import torch.utils.data
from torchvision.models.inception import inception_v3
import os
from scipy.stats import entropy
import argparse
import torch
import torchvision.transforms as transforms
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import ViT_H_14_Weights, vit_h_14
import numpy as np
# from clip import clip
from torchmetrics.functional import accuracy
from skimage.metrics import structural_similarity as ssim
# from scipy.spatial.distance import cosine
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor, Normalize
import argparse
import shutil
from pytorch_fid import fid_score
# from skimage import io, color
# torch.hub.set_dir("/leonardo_scratch/fast/IscrC_GenOpt/luigi/")
# from torchmetrics.image.kid import KernelInceptionDistance
# kid = KernelInceptionDistance(subset_size=50)
# Define the necessary transformations
transform = transforms.Compose([
    transforms.ToTensor()
])
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to("cuda")

# torch.hub.set_dir("/leonardo_scratch/fast/IscrC_GenOpt/luigi/")
# python evaluate_IS.py --root /mnt/media/luigi/model_out_CVPR_MULTISUB_FIXED_CAPTION/controlnet/generated 

# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def inception_score(args, resize=True):
    # Set up dtype
    device = torch.device(args.device)
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')

    files = readDir(args.root_IS)
    N = len(files)
    preds = np.zeros((N, 1000))
    if args.batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))

    for i in tqdm(range(0, N, args.batch_size)):
        start = i
        end = i + args.batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        y = get_pred(batch)
        preds[i:i + args.batch_size] = y

    assert args.batch_size > 0
    assert N > args.batch_size

    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    py = np.mean(preds, axis=0)  # marginal probability
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence

    mean_kl = np.mean(scores)
    inception_score = np.exp(mean_kl)

    return inception_score


def readDir(dirPath):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            f = os.path.join(dirPath, f)
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles.extend(subFiles)
            else:
                if "_gt" not in f:
                    allFiles.append(f)
        return allFiles
    else:
        return 'Error, not a dir'



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



def calc_lpips(img1_batch, img2_batch):
    #img1_batch = (img1_batch *2 ) - 1.0
    #img2_batch = (img2_batch *2 ) - 1.0
    # img1 = np.expand_dims(img1, axis=0)
    # img2 = np.expand_dims(img2, axis=0)
    return lpips(torch.FloatTensor(img1_batch).to("cuda"), torch.FloatTensor(img2_batch).to("cuda")).item()


# Function to load images from a directory in batches and convert them to tensors
def load_images_as_tensors_in_batches(directory, batch_size):
    filenames = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i + batch_size]
        image_tensors = []
        for filename in batch_filenames:
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
            image_tensor = transform(image)
            image_tensors.append(image_tensor)
        yield torch.stack(image_tensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--controlnet_path', type=str, default='picture-gene')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--GA', action='store_true')
    parser.add_argument('--guess', action='store_true', help="use guess folder")

    args = parser.parse_args()
    args.root_IS =  os.path.join(args.controlnet_path,'guess','generated') if args.guess else os.path.join(args.controlnet_path,'generated')
    IS = inception_score(args)
    print('The Inception Score is %.4f' % IS)
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    preprocess = weights.transforms()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    n_way = 50
    num_trials = 50
    top_k = 1

    acc_list = []
    gt_folder = os.path.join(args.controlnet_path,'guess','ground_truth') if args.guess else os.path.join(args.controlnet_path,'ground_truth')
    gene_folder= os.path.join(args.controlnet_path,'guess','generated')   if args.guess else os.path.join(args.controlnet_path,'generated') 
    
    gt_images_name = os.listdir(gt_folder)
    gt_images_name.sort()
    gt_image_num=0
    gn_imges_name = os.listdir(gene_folder)
    gn_imges_name.sort()
    if args.GA:
        for j in tqdm(range(0,len(gt_images_name), args.limit), total=len(gt_images_name)//args.limit):
            # print(gt_folder + gt_name)

            # Load GT image and the path of genetrated images
            real_image = Image.open(os.path.join(gt_folder,gt_images_name[j])).convert('RGB')

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
            for i in range(0,args.limit):
                # print(gene_folder + gn_imges_name[i])
                generated_image = Image.open(os.path.join(gene_folder,gn_imges_name[j+i])).convert('RGB')
                pred = preprocess(generated_image).unsqueeze(0).to("cuda")
                pred_out = model(pred).squeeze(0).softmax(0).detach()
                acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
                generated_image_list.append(generated_image)
                acc_list.append(acc)

            gt_image_num=gt_image_num+1

        print("MEAN GA:"+str(np.mean(acc_list)))


    temp_path1 = args.controlnet_path+'temppath'

    os.makedirs(temp_path1, exist_ok=True)

    for filename in os.listdir(gt_folder):
        shutil.copy(os.path.join(gt_folder, filename), os.path.join(temp_path1, filename))

    fid_value = fid_score.calculate_fid_given_paths([temp_path1, gene_folder], batch_size=50, device='cuda', dims=2048)

    print('FID:', fid_value)

    shutil.rmtree(temp_path1)

    # Batch size
    batch_size = 50

    # Initialize SSIM metric
    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # Process images in batches
    ssim_values,lpips_values,kid_values = [],[],[]
    for preds_batch, target_batch in tqdm(zip(load_images_as_tensors_in_batches(gene_folder, batch_size),
                                        load_images_as_tensors_in_batches(gt_folder, batch_size))):
        # Ensure preds and target have the same shape
        assert preds_batch.shape == target_batch.shape, "Preds and target must have the same shape"
        
        # # Calculate SSIM for the current batch
        # ssim_value = ssim(preds_batch, target_batch)
        # ssim_values.append(ssim_value)
        # Calculate LPIPS for the current batch
        lpips_value = calc_lpips(preds_batch, target_batch)
        lpips_values.append(lpips_value)
        
        # kid.update(target_batch.to(torch.uint8), real=True)
        # kid.update(preds_batch.to(torch.uint8), real=False)
        # kid_values.append(kid.compute()[0])
    # Calculate the mean SSIM value across all batches
    # mean_ssim = torch.mean(torch.tensor(ssim_values))
    mean_lpips = torch.mean(torch.tensor(lpips_values))
    # mean_kid = torch.mean(torch.tensor(kid_values))

    # print('Mean SSIM:', mean_ssim.item())
    print('Mean LPIPS:', mean_lpips.item())


    # print('Mean KID:', mean_kid.item())



