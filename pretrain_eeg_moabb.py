import random
import torch
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np 
import wandb
import yaml
from easydict import EasyDict as edict
from eeg_prepr_dreamdif import MOABB
sys.path.append("/home/beingfedericax/moab3.9/dn3/dn3")

from src.BENDR.dn3_ext import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.transforms.batch import RandomTemporalCrop

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDER model.")
    parser.add_argument('--config', default="configs/bendr_pretraining_moabb.yml", help="The DN3 config file to use.")
    parser.add_argument('--hidden-size', default=512, type=int, help="The hidden size of the encoder.")
    parser.add_argument('--resume', default=None, type=int, help="Whether to continue training the encoder from the "
                                                                 "specified epoch.")
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--no-save', action='store_true', help="Don't save checkpoints while training.")
    parser.add_argument('--no-save-epochs', action='store_true', help="Don't save epoch checkpoints while training")
    parser.add_argument('--dataset_folder', default='/mnt/media/lopez/moabb', help='Path to the dataset')
    parser.add_argument('--seed', default=0, type=int, help='Seed for reproducibility')
    return parser.parse_args()



def create_dataloaders(path,**kwargs):
    data = MOABB(root_dir=path, **kwargs)
    print("Num samples: ", len(data))
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size

    train_data, test_data = torch.utils.data.random_split(data, [train_size + val_size, test_size])
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader, val_loader

if __name__ == '__main__':
    args = parse_args()
    config = edict(yaml.safe_load(open(args.config)))
    wandb.init(project="DrEEam", settings=wandb.Settings(code_dir="."))
    wandb.run.log_code(".") # to save current code as an artifact in wandb
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update(config, allow_val_change=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    #per vedere se funziona la GPU
    print("Device: ", torch.cuda.get_device_name(0))

    # Set seed    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, test_loader, val_loader = create_dataloaders(args.dataset_folder, **config.dataset)
    print('Dataloaders created')

    print("Num training samples: ", len(train_loader.dataset))
    print("Num test samples: ", len(test_loader.dataset))
    print("Num val samples: ", len(val_loader.dataset))
    print("Dataloader shape: ", next(iter(train_loader))[0].shape, len(next(iter(train_loader))))

    encoder = ConvEncoderBENDR(in_features=config.dataset.in_channels, 
                               encoder_h=args.hidden_size, 
                               enc_downsample=config.bending_college_args.enc_downsample, 
                               enc_width=config.bending_college_args.enc_width,)
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=config.bending_college_args.layer_drop)
    # encoder.load('/home/luigi/Documents/DrEEam/src/DreamDiffusion/pretrains/models/BENDR_encoder.pt', strict=False)
    # encoder.freeze_features()
    # contextualizer.load('src/BENDR/contextualizer.pt', strict=False)
    # contextualizer.freeze_features()
    process = BendingCollegeWav2Vec(encoder, contextualizer, **config.bending_college_args)
    process.set_optimizer(torch.optim.Adam(process.parameters(), **config.optimizer_params)) 
    process.add_batch_transform(RandomTemporalCrop(config.augmentation_params.batch_crop_frac))

    def epoch_callback(metrics):
        # Save the model every epoch
        if not args.no_save:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            tqdm.write("Saving...")
            encoder.save(f'checkpoints/{wandb.run.name}_encoder.pt')
            contextualizer.save(f'checkpoints/{wandb.run.name}contextualizer.pt')

        # Log metrics
        print("metrics: ", metrics)
        wandb.log({f'epoch_val_{k}' if k != 'epoch' else k: v for k, v in metrics.items()})

    def log_callback(metrics):
        if metrics is not None and metrics['Accuracy'] > config.mask_threshold and \
                metrics['Mask_pct'] < config.mask_pct_max:
            process.mask_span = int(process.mask_span * config.mask_inflation) 
            wandb.log({'mask_span': process.mask_span})
        wandb.log({f'train_{k}' if k != 'iteration' else k: v for k, v in metrics.items()})  

    print('Training started')
    process.fit(train_loader, validation_dataset=val_loader, 
                num_workers=args.num_workers, resume_epoch=args.resume, 
                epoch_callback=epoch_callback, log_callback=log_callback,
                **config.training_params)
    
    print('Training completed, now evaluating...')
    # print(process.evaluate(test_loader))
    print(process.efficient_evaluate(test_loader))
    # Save checkpoint
    print('Saving checkpoint configuration...')
    #torch.save(process.state_dict(), '/home/beingfedericax/moab3.9/BENDR/checkpoints/checkpoint.pth')

    if not args.no_save:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        tqdm.write("Saving best model...")
        encoder.save(f'checkpoints/{wandb.run.name}_encoder_best_val.pt')
        contextualizer.save(f'checkpoints/{wandb.run.name}_contextualizer_best_val.pt')