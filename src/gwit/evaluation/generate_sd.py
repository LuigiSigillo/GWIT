import torch
from diffusers import StableDiffusionPipeline,UniPCMultistepScheduler
from datasets import load_dataset
import argparse
import torchvision.transforms as transforms

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    # torch_dtype=torch.float16,
    use_safetensors=True,
)
generator = torch.manual_seed(0)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--controlnet_path', type=str, default="/mnt/media/luigi/model_out_CVPR_MULTISUB_FIXED_CAPTION")
parser.add_argument('--limit', type=int, default=4)
args = parser.parse_args()  

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()
dset_name = "luigi-s/EEG_Image_CVPR_ALL_subj" #if not "single" in controlnet_path.lower() else "luigi-s/EEG_Image"
print(dset_name)
data_test = load_dataset(dset_name, split="test").with_format(type='torch')
data = data_test.filter(lambda x: x['subject'] == 4) if "single" in args.controlnet_path.lower() else data_test
 
# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)
# paper_classes =  ["lantern", "airliner", "panda"] if args.classes_to_find else None
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

classes_to_find_list = list(classes_to_find_dict.values())
limit = 4
import os
from tqdm import tqdm
num_samples = len(data_test)
all_samples = []
#check if folder exists
if not os.path.exists(f"{args.controlnet_path}/generated"):
    os.makedirs(f"{args.controlnet_path}/paper/", exist_ok=True)
    os.makedirs(f"{args.controlnet_path}/paper/generated",exist_ok=True)
    os.makedirs(f"{args.controlnet_path}/paper/ground_truth",exist_ok=True)
    controlnet_path = f"{args.controlnet_path}/paper"
for i in tqdm(range(0, num_samples)):
    found = False
    for c in classes_to_find_list:
        if c not in data[i]['caption']:
            continue
        else:
            found = True
            break
    if not found:
        continue
    gen_img_list = []
    
    control_image = data[i]['conditioning_image'].unsqueeze(0).to(torch.float16) #eeg DEVE essere #,128,512
    prompt = data[i]['caption'] #if "classifier" in controlnet_path.lower() else "image" #"image" #"real world image views or object" #data[i]['caption'] 
    # generate image
    images = pipe(prompt,num_inference_steps=20, generator=generator,num_images_per_prompt=limit).images
    # label = data_val[i]['caption'].replace("image of a", "")
    # image.save(f"{controlnet_path}/output_{i}_{label}.png")
    # print(data[i]['subject'])
    label = data[i]['caption'].replace("image of a", "")
    for j,image in enumerate(images):
        image.save(f"{controlnet_path}/generated/output_{i}_{j}_{label}.png")
        transforms.ToPILImage()(data[i]['image']).resize((512,512)).save(f"{controlnet_path}/ground_truth/gt_{i}_{j}_{label}.png")

