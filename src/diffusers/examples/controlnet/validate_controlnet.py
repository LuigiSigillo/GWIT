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


img_transform_test = transforms.Compose([
    # normalize, 
    transforms.Resize((512, 512)),   
])


base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "/leonardo_scratch/fast/IscrC_GenOpt/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/model_out_MULTISUB_LABEL_CAPTION/checkpoint-79500/controlnet"
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

dset_name = "luigi-s/EEG_Image_ALL_subj"
data_test = load_dataset(dset_name, split="validation").with_format(type='torch')

# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)

def generate(data, num_samples=10, limit=4, start=0, classes_to_find=None):
    all_samples = []
    for i in range(start, num_samples+start):
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
        prompt = "" #"real world image views or object" #data[i]['caption'] 
        # generate image
        images = pipe(
            prompt, num_inference_steps=20, generator=generator, image=control_image, 
            num_images_per_prompt=limit,
            subjects = torch.tensor([4]).unsqueeze(0) if not "ALL" in dset_name else data[i]['subject'].unsqueeze(0)
        ).images
        # label = data_val[i]['caption'].replace("image of a", "")
        # image.save(f"{controlnet_path}/output_{i}_{label}.png")
        # print(data[i]['subject'])

        gen_img_list = [transforms.ToTensor()(image).unsqueeze(0) for image in images]

        ground_truth = img_transform_test(data[i]['image']/255.).unsqueeze(0)
        gen_img_tensor = torch.cat(gen_img_list, dim=0)
        concatenated = torch.cat([ground_truth, gen_img_tensor], dim=0) # put groundtruth at first
        all_samples.append(concatenated)

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
        grid_image.save(f"{controlnet_path}/new_grid_image_{i}.png")
        # grid_image.show()

  
classes_to_find = ["lantern", "airliner", "panda"]
generate_grid(data_test, len(data_test), classes_to_find=classes_to_find)

# for i in range(len(data_test)):
#     print(data_test[i]['subject'], data_test[i]['caption'])