import torch
from tqdm import tqdm
from BENDR.dn3_ext import ConvEncoderBENDR
from eegdataset import EEGImageNet

if __name__ == "__main__":
    encoder_file = "encoder.pt"
    encoder = ConvEncoderBENDR(in_features=20, encoder_h=512)
    encoder.load(encoder_file, strict=True)
    encoder.freeze_features()
    encoder = encoder.to('cuda')

    # Prova
    # data = torch.randn(1, 20, 500).cuda()
    # latents = encoder(data.cuda())
    
    # Load the data
    data = EEGImageNet('/mnt/media/lopez/eeg_imagenet_10-20/mne_data', resample=False)
    data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
    latents = []
   
    for data in tqdm(data_loader, total=len(data_loader), desc="Computing latents"):
        latents.append(encoder(data.cuda()))
    
    latents = torch.cat(latents, dim=0)
    print("Latents shape: ", latents.shape)


encoder_file = "encoder.pt"
encoder = ConvEncoderBENDR(in_features=20, encoder_h=512)
encoder.load(encoder_file, strict=True)
encoder.freeze_features()
encoder = encoder.to('cuda')
def get_latents():
    pass