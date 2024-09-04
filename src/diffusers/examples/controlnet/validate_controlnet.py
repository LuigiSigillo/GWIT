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
controlnet_path = "/mnt/media/luigi/HDD_Documents/model_out_single_patient_caption"

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


data_test = load_dataset("luigi-s/EEG_Image", split="test").with_format(type='torch')

# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)

def generate(data_val, num_samples=10, limit=4):
    all_samples = []
    for i in range(0, num_samples):
        gen_img_list = []
        
        control_image = data_val[i]['conditioning_image'].unsqueeze(0).to(torch.float16) #eeg DEVE essere #,128,512
        prompt = "panda" #data_val[i]['caption'] 
        # generate image
        images = pipe(
            prompt, num_inference_steps=20, generator=generator, image=control_image, num_images_per_prompt=limit
        ).images
        # label = data_val[i]['caption'].replace("image of a", "")
        # image.save(f"{controlnet_path}/output_{i}_{label}.png")
        # print(prompt)

        gen_img_list = [transforms.ToTensor()(image).unsqueeze(0) for image in images]

        ground_truth = img_transform_test(data_val[i]['image']/255.).unsqueeze(0)
        gen_img_tensor = torch.cat(gen_img_list, dim=0)
        concatenated = torch.cat([ground_truth, gen_img_tensor], dim=0) # put groundtruth at first
        all_samples.append(concatenated)

    return all_samples

limit = 4
num_samples = 2
all_samples = generate(data_test,
                       num_samples , 
                       limit)
grid = rearrange(torch.stack(all_samples, dim=0), 'n b c h w -> (n b) c h w')

grid = make_grid(grid, padding=0, n_row=limit+1)
# Convert the grid to a PIL image
grid_image = transforms.ToPILImage()(grid)

# Save or display the image
grid_image.save(f"{controlnet_path}/grid_image.png")
# grid_image.show()

