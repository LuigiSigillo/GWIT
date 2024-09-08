import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torchvision.models import alexnet, resnet18, AlexNet_Weights, googlenet

np.random.seed(45)
torch.manual_seed(45)


class EEGFeatNet(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=128, num_layers=1):
        super(EEGFeatNet, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        # self.embedding  = nn.Embedding(num_embeddings=in_channels, embedding_dim=n_features)
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)
        # self.feat_layer = nn.Linear(in_features=self.hidden_size, out_features=n_features)
        # self.proj_layer = nn.Sequential(
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(in_features=n_features, out_features=projection_dim)
        #                                 )
    def forward(self, x):

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device) 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device)

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)
        x = F.normalize(x, dim=-1)
        # print(x.shape, feat.shape)
        return x#, feat

# class ImageFeatNet(nn.Module):

#     def __init__(self, projection_dim=128):
#         super(ImageFeatNet, self).__init__()
#         self.out        = projection_dim
#         # self.encoder    = resnet18(pretrained=True)
#         self.encoder    = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
#         for params in self.encoder.parameters():
#             params.requires_grad = False

#         self.regression = nn.Sequential(    
#                                             nn.Linear(1000, projection_dim, bias=True)
#                                         )
#         # self.encoder.fc = nn.Sequential(    
#         #                                     nn.Linear(self.encoder.fc.in_features, 1024),
#         #                                     nn.LeakyReLU(),
#         #                                     nn.Dropout(p=0.05),
#         #                                     nn.Linear(1024, projection_dim, bias=False)
#         #                                 )

#     def forward(self, x):
#         x = self.regression(self.encoder(x))
#         x = F.normalize(x, dim=-1)
#         return x

class ImageFeatNet(nn.Module):

    def __init__(self, projection_dim=128):
        super(ImageFeatNet, self).__init__()
        self.out        = projection_dim
        # self.encoder    = resnet18(pretrained=True)
        self.encoder    = googlenet(pretrained=True)
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.base_conv  = nn.Sequential(
                                            nn.Conv2d(in_channels=config.n_subjects+3, out_channels=3, kernel_size=3, padding=1),
                                            nn.ReLU()
                                        )

        self.encoder.fc = nn.Sequential(    
                                            nn.Linear(self.encoder.fc.in_features, 1024),
                                            nn.LeakyReLU(),
                                            nn.Dropout(p=0.05),
                                            nn.Linear(1024, projection_dim, bias=False)
                                        )

    def forward(self, x):
        x = self.base_conv(x)
        x = self.encoder(x)
        x = F.normalize(x, dim=-1)
        return x


if __name__ == '__main__':

    eeg   = torch.randn((512, 440, 128)).to(config.device)
    model = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim).to(config.device)
    proj, feat = model(eeg)
    print(proj.shape, feat.shape)
    # feat  = model(eeg)
    # print(feat.shape)