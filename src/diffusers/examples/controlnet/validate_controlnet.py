from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import torch
from datasets import load_dataset
from einops import rearrange, repeat
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--controlnet_path', type=str, default="/mnt/media/luigi/model_out_CVPR_MULTISUB_FIXED_CAPTION")
parser.add_argument('--limit', type=int, default=4)
parser.add_argument('--caption', action='store_true')
parser.add_argument('--classes_to_find', action='store_true', help="Don't generate plots")
parser.add_argument('--single_image_for_eval', action='store_true', help="Don't generate plots")
parser.add_argument('--guess', action='store_true', help="Don't generate plots")

args = parser.parse_args()  

# Get the current file path and directory
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Go up three levels from the current directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# print(base_dir)
# print(base_dir+"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40")
path_to_append = base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40" if "CVPR" in args.controlnet_path else base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz"
sys.path.append(path_to_append)
from network import EEGFeatNet
sys.path.append(base_dir+"/diffusers/src/dataset_EEG/")
if "CVPR" in args.controlnet_path :
    from name_map_ID import id_to_caption
else:
    from name_map_ID import id_to_caption_TVIZ as id_to_caption
model     = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in args.controlnet_path  else  \
            EEGFeatNet(n_classes=10, in_channels=14,\
                        n_features=128, projection_dim=128,\
                        num_layers=4).to("cuda")
model     = torch.nn.DataParallel(model).to("cuda")
import pickle

# Load the model from the file
pkl_path = base_dir+'/diffusers/src/dataset_EEG/knn_model.pkl' if "CVPR" in args.controlnet_path else base_dir+'/diffusers/src/dataset_EEG/knn_model_TVIZ.pkl'
with open(pkl_path, 'rb') as f:
    knn_cv = pickle.load(f)
ckpt_path = base_dir+"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth" if "CVPR" in args.controlnet_path \
    else base_dir+'/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz/EXPERIMENT_1/bestckpt/eegfeat_all_0.7212357954545454.pth' 
model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

def get_caption_from_classifier(eeg, labels):
    eeg =  torch.stack(eeg) if args.controlnet_path == "CVPR40" else torch.stack([torch.tensor(eeg_e) for eeg_e in eeg]) 
    x_proj = model(eeg.permute(0,2,1).to("cuda"))
    labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels]
    # Predict the labels
    predicted_labels = knn_cv.predict(x_proj.cpu().detach().numpy())
    captions = ["image of " + id_to_caption[label] for label in predicted_labels]
    return captions



def generate(data, num_samples=10, limit=4, start=0, classes_to_find=None, single_image_for_eval=False, 
             controlnet_path=None):
    all_samples = []
    #check if folder exists
    if not os.path.exists(f"{controlnet_path}/generated") and classes_to_find is None:
        os.makedirs(f"{controlnet_path}/generated",exist_ok=True)
        os.makedirs(f"{controlnet_path}/ground_truth",exist_ok=True)
    if classes_to_find is not None:
        os.makedirs(f"{controlnet_path}/paper/", exist_ok=True)
        os.makedirs(f"{controlnet_path}/paper/generated",exist_ok=True)
        os.makedirs(f"{controlnet_path}/paper/ground_truth",exist_ok=True)
        controlnet_path = f"{controlnet_path}/paper"
    if args.guess:
        os.makedirs(f"{controlnet_path}/guess/generated",exist_ok=True)
        os.makedirs(f"{controlnet_path}/guess/ground_truth",exist_ok=True)
        controlnet_path = f"{controlnet_path}/guess"
    else:
        controlnet_path = f"{controlnet_path}/paper"
    for i in tqdm(range(start, num_samples+start)):
        found = False
        if classes_to_find is not None:
            for c in classes_to_find:
                if c not in data[i]['caption']:
                    continue
                else:
                    found = True
                    break
            if not found:
                continue
        gen_img_list = []
        
        control_image = data[i]['conditioning_image'].unsqueeze(0).to(torch.float16) #eeg DEVE essere #,128,512
        #TODO mettere classificatore in effetti 
        #eeg_key = "conditioning_pixel_values" if "CVPR" in args.controlnet_path else "eeg_no_resample"
        #prompt = get_caption_from_classifier(data[i][eeg_key], data[i]["label"]) 
        
        prompt = data[i]['caption'] if "classifier" in controlnet_path.lower() else "image" #"image" #"real world image views or object" #data[i]['caption'] 
        prompt = data[i]['caption'] if args.caption else prompt
        # generate image
        images = pipe(
            prompt, num_inference_steps=20, generator=generator, image=control_image, 
            num_images_per_prompt=limit,
            subjects = data[i]['subject'].unsqueeze(0),
            guess_mode=args.guess,
            guidance_scale=4.0 if args.guess else 7.5, #default value
        ).images
        # label = data_val[i]['caption'].replace("image of a", "")
        # image.save(f"{controlnet_path}/output_{i}_{label}.png")
        # print(data[i]['subject'])
        if not single_image_for_eval:
            gen_img_list = [transforms.ToTensor()(image).unsqueeze(0) for image in images]

            ground_truth = img_transform_test(data[i]['image']/255.).unsqueeze(0)
            gen_img_tensor = torch.cat(gen_img_list, dim=0)
            concatenated = torch.cat([ground_truth, gen_img_tensor], dim=0) # put groundtruth at first
            all_samples.append(concatenated)
        else:
            label = data[i]['caption'].replace("image of a", "")
            for j,image in enumerate(images):
                image.save(f"{controlnet_path}/generated/output_{i}_{j}_{label}.png")
                transforms.ToPILImage()(data[i]['image']).resize((512,512)).save(f"{controlnet_path}/ground_truth/gt_{i}_{j}_{label}.png")
    return all_samples


def generate_grid(data_test, num_samples=10, classes_to_find=None):
    limit = 7
    for i in range(0,num_samples,10):
        all_samples = generate(data_test,
                            num_samples=10 , 
                            limit=limit,
                            start=i,
                            classes_to_find=classes_to_find)
        if len(all_samples) == 0:
            continue
        grid = rearrange(torch.stack(all_samples, dim=0), 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, padding=0, n_row=limit+1)
        # Convert the grid to a PIL image
        grid_image = transforms.ToPILImage()(grid)

        # Save or display the image
        grid_image.save(f"{args.controlnet_path}/new_grid_image_{i}.png" if classes_to_find is not None else f"{args.controlnet_path}/grid_image_{i}.png")
        # grid_image.show()

  

img_transform_test = transforms.Compose([
    # normalize, 
    transforms.Resize((512, 512)),   
])


base_model_path = "stabilityai/stable-diffusion-2-1-base"
# controlnet_path = "/mnt/media/luigi/model_out_CVPR_MULTISUB_FIXED_CAPTION"
controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

dset_name = "luigi-s/EEG_Image_CVPR_ALL_subj" if "CVPR" in args.controlnet_path else  "luigi-s/EEG_Image_TVIZ_ALL_subj" #if not "single" in controlnet_path.lower() else "luigi-s/EEG_Image"
print(dset_name)
data_test = load_dataset(dset_name, split="test", cache_dir="/mnt/media/luigi/").with_format(type='torch')
data_test = data_test.filter(lambda x: x['subject'] == 4) if "single" in args.controlnet_path.lower() else data_test
 
# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)
paper_classes =  ["lantern", "airliner", "panda"] if args.classes_to_find else None
# generate_grid(data_test, len(data_test)-1, classes_to_find=classes_to_find)
classes_to_find_dict = {11: 'lycaenid, lycaenid butterfly', 
                   21: 'capuchin, ringtail, Cebus capucinus', 
                   8: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 
                   6: 'airliner', 
                   10: 'canoe', 
                #    15: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
                #    5: 'coffee mug', 
                #    24: 'convertible', 
                   37: 'electric locomotive',
                #    25: 'folding chair', 
                   14: "jack-o'-lantern",
                #    27: 'mitten', 
                #    1: 'parachute, chute', 
                   38: 'radio telescope, radio reflector', 
                   7: 'revolver, six-gun, six-shooter',
                   9: 'daisy'
                   }

classes_to_find_list = list(classes_to_find_dict.values()) if args.classes_to_find else None
# for i in range(len(data_test)):
#     print(data_test[i]['subject'], data_test[i]['caption'])
print(len(data_test))
if not args.single_image_for_eval:
    args.limit = 7 #to create the grid
    generate_grid(data_test, len(data_test)-1, classes_to_find=classes_to_find_list)

else:
    all_samples = generate(data_test,
                        num_samples=len(data_test) , 
                        limit=args.limit,
                        start=0,
                        classes_to_find=classes_to_find_list, 
                        single_image_for_eval=args.single_image_for_eval,
                        controlnet_path=args.controlnet_path)