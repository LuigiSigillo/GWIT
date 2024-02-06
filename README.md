# DrEEam
## Environment setup

Create and activate conda environment named ```dreamdiffusion``` from the ```env.yaml```
```sh
conda env create -f src/DreamDiffusion/env.yaml
conda activate dreamdiffusion
pip install src/dn3-0.2-aplha/.
```
```sh
python3 src/DreamDiffusion/code/eeg_ldm.py --dataset EEG  --num_epoch 300 --batch_size 4 --pretrain_mbm_path src/DreamDiffusion/pretrains/encoder.pth
```