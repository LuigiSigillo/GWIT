import os
import random
from pandas import DataFrame
import torch
import tqdm
import argparse
import numpy as np 
import yaml
from easydict import EasyDict as edict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange

from pretrain_eeg_moabb import create_dataloaders
from eegdataset import preprocess_EEG_data
from src.DreamDiffusion.code.dataset import EEGDataset, Splitter
from src.BENDR.dn3_ext import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer

# Functions needed for Spampy
def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDER model.")
    parser.add_argument('--dataset', default='MOABB', help="The dataset to use.")
    parser.add_argument('--config', default="configs/bendr_pretraining_moabb.yml", help="The DN3 config file to use.")
    parser.add_argument('--hidden-size', default=512, type=int, help="The hidden size of the encoder.")
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--dataset_folder', default='/mnt/media/lopez/moabb', help='Path to the dataset')
    parser.add_argument('--seed', default=0, type=int, help='Seed for reproducibility')
    parser.add_argument('--run_name', type=str, help='Name of the run')
    parser.add_argument('--enc_downsample', nargs='+', type=int, default=[3,2], help='Encoder downsample values')
    parser.add_argument('--mask_span', type=int, default=5, help='Mask span value')
    parser.add_argument('--no_plot', action='store_true', help="Don't generate plots")
    parser.add_argument('--no_evaluate', action='store_true', help="Only generate figures")
    return parser.parse_args()

def evaluate(process: BendingCollegeWav2Vec, dataloader, save_path, plot=True, evaluate=True, dataset='MOABB'):
    process.train(False)
    data_iterator = iter(dataloader)
    pbar = tqdm.trange(len(dataloader), desc="Evaluating")
    log = list()
    with torch.no_grad():
        for iteration in pbar:
            inputs = process._get_batch(data_iterator)
            outputs = process.forward(*inputs)
            metrics = process.calculate_metrics(inputs, outputs)
            metrics['loss'] = process.calculate_loss(inputs, outputs).item()
            metrics['iteration'] = iteration

            _, bendr, mask, reconstructed_bendr = outputs
            metrics['corr'] = np.corrcoef(bendr.cpu().numpy().flatten(), 
                                            reconstructed_bendr[:, :, 1:].cpu().numpy().flatten())[0, 1]
            log.append(metrics)

            ### Plot some samples ###
            if iteration == 0 and plot:
                print("Plotting samples")
                masked_bendr = bendr.clone()
                # print positions where mask is true
                print(np.where(mask.cpu().numpy()))
                masked_bendr.transpose(2, 1)[mask] = np.nan # for plotting purposes

                # Number of random samples to select
                num_samples = 3

                # Select random samples from the batch
                indices = np.random.choice(bendr.shape[0], size=num_samples, replace=False)

                # Index into your data
                input_samples = inputs[0][indices].cpu().numpy()
                bendr_samples = bendr[indices].cpu().numpy()
                masked_bendr_samples = masked_bendr[indices].cpu().numpy()
                reconstructed_bendr_samples = reconstructed_bendr[indices][:, :, 1:].cpu().numpy()

                # Number of random channels to select
                num_channels = 3

                # Select random channels
                channels = np.random.choice(bendr.shape[1], size=num_channels, replace=False)
                print("channels: ", channels)

                for i, index in enumerate(indices):
                    fig, axs = plt.subplots(num_channels, 4, figsize=(30, 10))
                    fig.suptitle(f'Sample {index}')

                    for j, channel in enumerate(channels):
                        input_samples_channel = input_samples[i, 0].squeeze()
                        bendr_samples_channel = bendr_samples[i, channel].squeeze()
                        masked_bendr_samples_channel = masked_bendr_samples[i, channel].squeeze()
                        reconstructed_bendr_samples_channel = reconstructed_bendr_samples[i, channel].squeeze()

                        axs[j, 0].plot(input_samples_channel)
                        axs[j, 0].set_title('Original EEG')
                        axs[j, 0].set_ylabel(f'Channel {channel+1}', fontsize=14)

                        axs[j, 1].plot(bendr_samples_channel)
                        axs[j, 1].set_title('Bendr')

                        axs[j, 2].plot(masked_bendr_samples_channel)
                        axs[j, 2].set_title('Masked Bendr')

                        axs[j, 3].plot(reconstructed_bendr_samples_channel)
                        axs[j, 3].set_title('Reconstructed Bendr')

                    plt.savefig(f"{save_path}/{dataset}_sample_{index}.png")

                # Only plot
                if not evaluate:
                    return

    metrics = DataFrame(log)
    metrics = metrics.mean().to_dict()
    metrics.pop('iteration', None)
    return metrics

if __name__ == '__main__':
    args = parse_args()
    config = edict(yaml.safe_load(open(args.config)))

    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    print("Device: ", torch.cuda.get_device_name(0))

    # Set seed for dataset reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.dataset == 'MOABB':
        train_loader, test_loader, val_loader = create_dataloaders(args.dataset_folder, **config.dataset)
    elif args.dataset == 'Spampy':
        img_transform_test = transforms.Compose([
            normalize, transforms.Resize((512, 512)), 
            channel_last
        ])
        dataset_test = EEGDataset('/home/luigi/Documents/DrEEam/dataset/eeg_5_95_std.pth', img_transform_test, 4)
        split_test = Splitter(dataset_test, 
                              split_path='/home/luigi/Documents/DrEEam/dataset/block_splits_by_image_single.pth', 
                              split_num=0, split_name='test', subject=4)
        # dataset_train = EEGDataset('/home/luigi/Documents/DrEEam/dataset/eeg_5_95_std.pth', img_transform_test, 4)
        # split_train = Splitter(dataset_train, 
        #                       split_path='/home/luigi/Documents/DrEEam/dataset/block_splits_by_image_single.pth', 
        #                       split_num=0, split_name='train', subject=4)
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            batch = torch.utils.data.dataloader.default_collate(batch)['eeg']
            # batch = preprocess_EEG_data(batch, resample=False) # For original BENDR
            return [batch]
        test_loader = DataLoader(split_test, batch_size=16, shuffle=False, num_workers=6, collate_fn=collate_fn)
        print("shape: ", next(iter(test_loader))[0].shape)

    encoder = ConvEncoderBENDR(in_features=config.dataset.in_channels, 
                               encoder_h=args.hidden_size, 
                               enc_downsample=args.enc_downsample, 
                               enc_width=args.enc_downsample,)
    # encoder = ConvEncoderBENDR(in_features=20, encoder_h=args.hidden_size) # For original BENDR
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=config.bending_college_args.layer_drop)
    encoder.load(f'checkpoints/{args.run_name}_encoder_best_val.pt')
    encoder.freeze_features()
    contextualizer.load(f'checkpoints/{args.run_name}_contextualizer_best_val.pt')
    contextualizer.freeze_features()

    # For original BENDR
    # encoder.load('/home/luigi/Documents/DrEEam/src/DreamDiffusion/pretrains/models/BENDR_encoder.pt', strict=False)
    # encoder.freeze_features()
    # contextualizer.load('src/BENDR/contextualizer.pt', strict=False)
    # contextualizer.freeze_features()

    process = BendingCollegeWav2Vec(encoder, contextualizer, mask_span=args.mask_span, **config.bending_college_args)
    
    if not os.path.exists(f"figures/{args.run_name}"):
        os.makedirs(f"figures/{args.run_name}")

    print(evaluate(process, test_loader, f"figures/{args.run_name}", plot=not args.no_plot, evaluate=not args.evaluate, dataset=args.dataset))

    