# GWIT - Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models.
![visitors](https://visitor-badge.laobi.icu/badge?page_id=luigisigillo/GWIT)
[![Paper](https://img.shields.io/badge/arXiv-2405.02771-blue)](https://arxiv.org/abs/2410.02780)

Official PyTorch repository for GWIT, Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models.
## Abstract :bookmark_tabs:
Generating images from brain waves is gaining increasing attention due to its potential to advance brain-computer interface (BCI) systems by understanding how brain signals encode visual cues. Most of the literature has focused on fMRI-to-Image tasks as fMRI is characterized by high spatial resolution. However, fMRI is an expensive neuroimaging modality and does not allow for real-time BCI. On the other hand, electroencephalography (EEG) is a low-cost, non-invasive, and portable neuroimaging technique, making it an attractive option for future real-time applications. Nevertheless, EEG presents inherent challenges due to its low spatial resolution and susceptibility to noise and artifacts, which makes generating images from EEG more difficult. In this paper, we address these problems with a streamlined framework based on the ControlNet adapter for conditioning a latent diffusion model (LDM) through EEG signals. We conduct experiments and ablation studies on popular benchmarks to demonstrate that the proposed method beats other state-of-the-art models. Unlike these methods, which often require extensive preprocessing, pretraining, different losses, and captioning models, our approach is efficient and straightforward, requiring only minimal preprocessing and a few components. 


[Eleonora Lopez](), [Luigi Sigillo](https://luigisigillo.github.io/), [Federica Colonnese](), [Massimo Panella](https://massimopanella.site.uniroma1.it/) and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/home)

[ISPAMM Lab](https://ispamm.it/) and [NESYA Lab](https://sites.google.com/view/nesya) , Sapienza University of Rome 
## Model Architecture :clapper:
<img src="assets/architecture.png" width="500px"/>

## Update
- **xx.xx.2025**: 
- **xx.xx.2024**: Checkpoints are released.
- **xx.xx.2024**: Repo is released.

[<img src="assets/results.png" />]() 

For more evaluation, please refer to our [paper](https://arxiv.org/abs/2410.02780) for details.

## How to run experiments :computer:

```bash
pip install src/diffusers
```

```bash
pip install transformers accelerate xformers==0.0.16 wandb numpy==1.26.4 datasets torchvision==0.14.1
```
#### Train
To launch the training of the model, you can use the following command, you need to change the output_dir:
```bash
accelerate launch src/gwit/train_controlnet.py --caption_from_classifier --subject_num=4 --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base --output_dir=output/model_out_CVPR_SINGLE_SUB_CLASSIFIER_CAPTION --dataset_name=luigi-s/EEG_Image_CVPR_ALL_subj --conditioning_image_column=conditioning_image --image_column=image --caption_column=caption --resolution=512 --learning_rate=1e-5 --train_batch_size=8 --num_train_epochs=50 --tracker_project_name=controlnet --enable_xformers_memory_efficient_attention --checkpointing_steps=1000 --validation_steps=500 --report_to wandb --validation_image ./using_VAL_DATASET_PLACEHOLDER.jpeg --validation_prompt "we are using val dataset hopefuly"
```
You can change the dataset using one of, with the dataset_name parameter: 
- luigi-s/EEG_Image_CVPR_ALL_subj
- luigi-s/EEG_Image_TVIZ_ALL_subj

#### Generate directly
Request access to the pretrained models from [Google Drive]().

To launch the generation of the images from the model, you can use the following command, you need to change the output_dir:
```bash
python src/gwit/validate_controlnet.py --controlnet_path=output/model_out_CVPR_SINGLE_SUB_CLASSIFIER_CAPTION/checkpoint-24000/controlnet/ --caption --single_image_for_eval --guess
```

#### Evaluation directly
Request access to the pretrained models from [Google Drive]().

To launch the testing of the model, you can use the following command, you need to change the output_dir:
```bash
python gwit/evaluation/evaluate.py --controlnet_path=output/model_out_CVPR_SINGLE_SUB_CLASSIFIER_CAPTION/checkpoint-24000/controlnet/ --caption --single_image_for_eval --guess
```
#### Validation 
todo
```bash
python src/gwit/evaluation.py --controlnet_path=output/model_out_CVPR_SINGLE_SUB_CLASSIFIER_CAPTION/checkpoint-24000/controlnet/ --caption --single_image_for_eval --guess
```

## Dataset
The dataset used are hosted on huggingface: 

- [ImageNetEEG](https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj)
- [Thoughtviz](https://huggingface.co/datasets/luigi-s/EEG_Image_TVIZ_ALL_subj)


## Cite
Please cite our work if you found it useful:
```
@misc{lopez2024guessithinkstreamlined,
      title={Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models}, 
      author={Eleonora Lopez and Luigi Sigillo and Federica Colonnese and Massimo Panella and Danilo Comminiello},
      year={2024},
      eprint={2410.02780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02780}, 
}
```


## Acknowledgement

This project is based on [diffusers](https://github.com/huggingface/diffusers). Thanks for their awesome work.
