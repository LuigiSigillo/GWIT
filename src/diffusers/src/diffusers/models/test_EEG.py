from torch import nn
import torch
import torch.nn.functional as F

class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        #Extract Dimensions:
        _, C, D = self.weights.shape
        #
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

def test(CONDITIONING_CHANNELS = 128,
         N_SUBJECTS = 7, 
         CONDITIONING_EMBEDDING_CHANNELS = 320):
    
    # WHAT WE RECEIVE AS INPUTS
    conditioning = torch.randn(4, 128, 512)
    subjects = torch.tensor([4]*conditioning.shape[0])
    
    #what we can change
    block_strides = (1,1,1,1)
    # block_out_channels = (320, 640, 1280, 2560)
    block_out_channels = (128,128,128,128)

    subj_layers = SubjectLayers(CONDITIONING_CHANNELS, CONDITIONING_CHANNELS, N_SUBJECTS)
    conv_in = nn.Conv1d(CONDITIONING_CHANNELS, block_out_channels[0], kernel_size=1, stride=1)

    blocks = nn.ModuleList([])

    for i in range(len(block_out_channels) - 1):
        channel_in = block_out_channels[i]
        channel_out = block_out_channels[i + 1]
        blocks.append(nn.Conv1d(channel_in, channel_in, kernel_size=1,stride=block_strides[i + 1]))
        blocks.append(nn.Conv1d(channel_in, channel_out, kernel_size=1, stride=block_strides[i + 1]))


    conditioning = subj_layers(conditioning, subjects)
    embedding = conv_in(conditioning)
    embedding = F.silu(embedding)
    for block in blocks:
        embedding = block(embedding)
        embedding = F.silu(embedding)

    embedding = torch.cat([embedding]*20, dim=0)
    embedding = embedding.reshape(conditioning.shape[0],
                                    CONDITIONING_EMBEDDING_CHANNELS,
                                    64,
                                    64)
    embedding = embedding.permute(0, 1, 3, 2)
    # embedding = self.conv_out(embedding)
    # Pad to (4, 320, 64, 64)
    padding = (0, 64-embedding.shape[3], 0, 0)  # Pad the last dimension to 64
    embedding = nn.functional.pad(embedding, padding, mode='constant', value=0)
    return embedding

# embedding = test()
# print(embedding.shape)
# assert embedding.shape == torch.Size([4, 320, 64, 64])
# import sys
# sys.path.append('/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEGStyleGAN-ADA_CVPR40/')
# from network import EEGFeatNet
# feat_dim = 128
# projection_dim = 128
# num_layers = 4
# device = "cuda"
# eeg_model = EEGFeatNet(n_features=feat_dim, projection_dim=projection_dim, num_layers=num_layers).to(device)
# eeg_model = torch.nn.DataParallel(eeg_model).to(device)
# eegckpt   = '/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEGStyleGAN-ADA_CVPR40/eegbestckpt/eegfeat_lstm_all_0.9665178571428571.pth'
# eegcheckpoint = torch.load(eegckpt, map_location=device)
# eeg_model.load_state_dict(eegcheckpoint['model_state_dict'])
from datasets import load_dataset
data = load_dataset('luigi-s/EEG_Image', split='test').with_format('torch')

i = 2048
embedding_dim  = 256
projection_dim = 256
num_layers = 1
device = "cuda"
model_path = "/home/luigi/Documents/DrEEam/src/diffusers/src/diffusers/models/checkpoints_EEG_CLIP/clip_"
model_path_c = model_path + str(i) + ".pth"

checkpoint = torch.load(model_path_c, map_location=device)
import sys
sys.path.append("/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEGClip/")
from CLIPModel import CLIPModel
from EEG_encoder import EEG_Encoder
eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)
# eegckpt   = '/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEGStyleGAN-ADA_CVPR40/eegbestckpt/eegfeat_lstm_all_0.9665178571428571.pth'
# eeg_embedding.load_state_dict(torch.load(eegckpt))
from torchvision.models import resnet50
image_embedding = resnet50(pretrained=False).to(device)
num_features = image_embedding.fc.in_features

image_embedding.fc = nn.Sequential(
    nn.ReLU(),
    nn.Linear(num_features, embedding_dim, bias=False)
)

image_embedding.fc.to(device)

model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

model = model.text_encoder

for param in model.parameters():
    param.requires_grad = True

new_layer = nn.Sequential(
    nn.Linear(embedding_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 40),
    nn.Softmax(dim=1)
)

model.fc = nn.Sequential(
    model.fc,
    new_layer
)

model = model.to(device)
model = torch.nn.DataParallel(model).to(device)
#checkpoint con cose di inception strane
# model.load_state_dict(torch.load("/home/luigi/Documents/DrEEam/src/diffusers/src/diffusers/models/checkpoints_EEG_CLIP/eegfeat_all_0.6875.pth")['model_state_dict'])
#checkpoint LSTM encoder EEG
# model.load_state_dict(torch.load("/home/luigi/Documents/DrEEam/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/finetune_bestckpt/eegfeat_all_0.9833920483140413.pth")['model_state_dict'])

model.eval()

for d in data:
    x_val_eeg = d['conditioning_image'].unsqueeze(0).permute(0,2,1).to(device)
    # print(x.shape)

    outputs = model(x_val_eeg)
    _, predicted = torch.max(outputs.data, 1)
    # correct = (predicted == labels_val).sum().item()
    # print('Accuracy of the network %d %%' % (100 * correct / 1994))
    # val_acc = 100 * correct / 1994
    print(predicted, d['caption'], d['label_folder'])
