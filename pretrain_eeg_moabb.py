import torch
import tqdm
import argparse
import sys
import os
import numpy as np 
import wandb
import yaml
from easydict import EasyDict as edict
from eeg_prepr_dreamdif import EEG_preprocessing
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
    parser.add_argument('--dataset_folder', default='/mnt/media/lopez/dataset_eeg_final', help='Path to the dataset')
    return parser.parse_args()



def create_dataloaders(path):
    data = EEG_preprocessing(root_dir=path)
    print('Preprocessing Completed!')
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
    print(device)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())    
    print(torch.cuda.get_device_name(0))

    train_loader, test_loader, val_loader = create_dataloaders(args.dataset_folder)
    print('Dataloaders created')

    #train_loader = train_loader.to(device)
    #test_loader = test_loader.to(device)
    #val_loader = val_loader.to(device)

    encoder = ConvEncoderBENDR(in_features=128, encoder_h=args.hidden_size)
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=config.bending_college_args.layer_drop)
    process = BendingCollegeWav2Vec(encoder, contextualizer, **config.bending_college_args)
    process.set_optimizer(torch.optim.Adam(process.parameters(), **config.optimizer_params)) 
    process.add_batch_transform(RandomTemporalCrop(config.augmentation_params.batch_crop_frac))

    def epoch_callback(metrics):
        # Save the model every epoch
        if not args.no_save:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            tqdm.tqdm.write("Saving...")
            encoder.save(f'checkpoints/{wandb.run.name}_encoder.pt')
            contextualizer.save(f'checkpoints/{wandb.run.name}contextualizer.pt')

        # Log metrics
        print("metrics: ", metrics)
        wandb.log(metrics)    
    
    print('Training started')
    process.fit(train_loader, validation_dataset=val_loader, 
                num_workers=args.num_workers, resume_epoch=args.resume, epoch_callback=epoch_callback,
                **config.training_params)
    
    print('Training completed, now evaluating...')
    print(process.evaluate(test_loader))
    # Save checkpoint
    print('Saving checkpoint configuration...')
    #torch.save(process.state_dict(), '/home/beingfedericax/moab3.9/BENDR/checkpoints/checkpoint.pth')

    if not args.no_save:
        if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
        tqdm.tqdm.write("Saving best model...")
        encoder.save(f'checkpoints/{wandb.run.name}_encoder_best_val.pt')
        contextualizer.save(f'checkpoints/{wandb.run.name}_contextualizer_best_val.pt')