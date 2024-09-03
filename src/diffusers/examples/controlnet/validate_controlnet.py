from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import torch
from datasets import load_dataset

base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "/home/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/model_out"

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


data_val = load_dataset("luigi-s/EEG_Image", split="validation").with_format(type='torch')

# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
generator = torch.manual_seed(0)

for i in range(0, 10):
    control_image = data_val[i]['conditioning_image'].unsqueeze(0).to(torch.float16) #eeg DEVE essere #,128,512
    prompt = "" #data_val[i]['caption'] 
    # generate image
    image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=control_image
    ).images[0]
    label = prompt.replace("image of a", "")
    image.save(f"/home/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/model_out/output_{i}_{prompt}.png".format(i, label))
    # print(prompt)
