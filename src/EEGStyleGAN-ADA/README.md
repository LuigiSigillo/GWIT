# EEGStyleGAN-ADA
Pytorch code of paper "Learning Robust Deep Visual Representations from EEG Brain Recordings." [Accepted in WACV 2024]  [[Paper](https://arxiv.org/abs/2310.16532)]

1. Anaconda environment yml file is present in the Anaconda folder. Use it to create a conda environment.
2. For StyleGAN-ADA, we have used the official Pytorch implementation. [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch)
3. Command to train the GAN is mentioned in the Txt file.
4. EEGClip code is unstructured.
5. Checkpoints:
   * EEGStyleGAN-ADA-CVPR40 [[link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EXn-8R80rxtHjlMCzPfhL9UBj80opHXyq3MnBBXXE6IsQw?e=Xbt2zO)]
   * EEGStyleGAN-ADA-thoughtviz [[link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EcfBxiKOk1NEqMDvE7juYlYB8wb0mKkWcc1RQmb9Ze8TUQ?e=HrzBsU)]
   * EEGClip [[link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EZW2pQA7I4VNsAxLVBrY8CkBLtO0Bg3ho7lsfTXlJVfyfQ?e=ygVt8u)] (raw EEG)
   * Image2Image (imageckpt) [[link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EQuJKUdXz8lGn04O2KSwsAUBUdsL-rjj0FdDwNH1Z8V9jw?e=716Xuq)]

## Notes

* Previous work EEG2IMAGE: Image Reconstruction from EEG Brain Signals [[Link](https://arxiv.org/abs/2302.10121)] (ICASSP 2023).
* For EEGClip please check this issue [[link](https://github.com/prajwalsingh/EEGStyleGAN-ADA/issues/9#issuecomment-1969719968)].

## Config

<pre>
conda create -n to1.7 anaconda python=3.8
conda activate to1.7
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.5.4.58 opencv-contrib-python==4.5.4.58
pip install natsort
</pre>

## Introduction

Decoding the human brain has been a hallmark of neuroscientists and Artificial Intelligence researchers alike. Reconstruction of visual images from brain Electroencephalography (EEG) signals has garnered a lot of interest due to its applications in brain-computer interfacing. This study proposes a two-stage method where the first step is to obtain EEG-derived features for robust learning of deep representations and subsequently utilize the learned representation for image generation and classification. We demonstrate the generalizability of our feature extraction pipeline across three different datasets using deep-learning architectures with supervised and contrastive learning methods. We have performed the zero-shot EEG classification task to support the generalizability claim further. We observed that a subject invariant linearly separable visual representation was learned using EEG data alone in an unimodal setting that gives better k-means accuracy as compared to a joint representation learning between EEG and images. Finally, we propose a novel framework to transform unseen images into the EEG space and reconstruct them with approximation, showcasing the potential for image reconstruction from EEG signals. Our proposed image synthesis method from EEG shows $62.9\%$ and $36.13\%$ inception score improvement on the EEGCVPR40 and the Thoughtviz datasets, which is better than state-of-the-art performance in GAN.

| Feature Extraction and Image Synthesis Architecture  |
|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/architecture.png" width="1024px" height="512px"/>  |

| Learned EEG Space using Triplet Loss with LSTM and CNN Architecture  |
|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/eegspace.png" width="1024px" height="512px"/>  |

| CVPR40 Dataset (40 Classes)  | ThoughtViz Dataset (10 Classes) |
|---|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/seed0000-min.png" width="512px" height="512px"/>  | <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/fakes005725-min.png" width="512px" height="512px"/>  |

## References

* StyleGAN2-ADA [[Link](https://github.com/NVlabs/stylegan2-ada-pytorch)]
