import torch
import tqdm
import argparse
import sys
import os
import numpy as np 
from eeg_prepr_dreamdif import EEG_preprocessing
sys.path.append("/home/beingfedericax/moab3.9/dn3/dn3")

from dn3_ext import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.configuratron import ExperimentConfig
from dn3.transforms.instance import To1020
from dn3.transforms.batch import RandomTemporalCrop
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDER model.")
    #parser.add_argument('--config', default="configs/pretraining.yml", help="The DN3 config file to use.")
    parser.add_argument('--hidden-size', default=960, type=int, help="The hidden size of the encoder.")
    parser.add_argument('--resume', default=None, type=int, help="Whether to continue training the encoder from the "
                                                                 "specified epoch.")
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--no-save', action='store_true', help="Don't save checkpoints while training.")
    parser.add_argument('--no-save-epochs', action='store_true', help="Don't save epoch checkpoints while training")
    return parser.parse_args()



def create_dataloaders(path):
    eeg_tot = []
    data = EEG_preprocessing(root_dir=path)
    print('Preprocessing Completed!')
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size

    train_data, test_data = torch.utils.data.random_split(data, [train_size + val_size, test_size])
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=300, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=300, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    return train_loader, test_loader, val_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    #per vedere se funziona la GPU
    print(device)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())    
    print(torch.cuda.get_device_name(0))

    args = parse_args()
    dataset = '/home/beingfedericax/moab3.9/dataset_eeg_final'
    train_loader, test_loader, val_loader = create_dataloaders(dataset)
    print('Dataloaders created')

    #train_loader = train_loader.to(device)
    #test_loader = test_loader.to(device)
    #val_loader = val_loader.to(device)

    encoder = ConvEncoderBENDR(in_features=128, encoder_h=960)
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=0.01)
    process = BendingCollegeWav2Vec(encoder, contextualizer, mask_rate=0.065, mask_span=2, layer_drop=0.01, temp=0.1, encoder_grad_frac=0.1, num_negatives=20, enc_feat_l2=1.0)
    process.set_optimizer(torch.optim.Adam(process.parameters(), lr=0.00002, weight_decay=0.01, betas=[0.9, 0.98])) 
    process.add_batch_transform(RandomTemporalCrop(max_crop_frac=0.05))
    print('Training started')
    process.fit(train_loader, epochs=1, num_workers=args.num_workers,
            validation_dataset=val_loader, resume_epoch=args.resume, validation_interval= 300, train_log_interval=300, batch_size=300, warmup_frac=0.05)
    print('Training completed, now evaluating...')
    print(process.evaluate(test_loader))
    # Save checkpoint
    print('Saving checkpoint configuration...')
    #torch.save(process.state_dict(), '/home/beingfedericax/moab3.9/BENDR/checkpoints/checkpoint.pth')

    if not args.no_save:
        tqdm.tqdm.write("Saving best model...")
        encoder.save('/home/beingfedericax/moab3.9/BENDR/checkpoints/encoder_best_val.pt')
        contextualizer.save('/home/beingfedericax/moab3.9/BENDR/checkpoints/contextualizer_best_val.pt')