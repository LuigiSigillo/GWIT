import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torchvision.models import googlenet

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
        return x


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


class EEGCNNFeatNet(nn.Module):
    def __init__(self, input_shape=(1, 440, 128), n_features=128, projection_dim=128, num_filters=[128, 256, 512, 1024], kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2], padding=[1, 1, 1, 1]):
        super(EEGCNNFeatNet, self).__init__()

        # Define the convolutional layers
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for i, out_channels in enumerate(num_filters):
            self.layers.append( 
                                nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i], bias=False),\
                                    # nn.BatchNorm2d(out_channels),\
                                    nn.InstanceNorm2d(out_channels),\
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.05),
                                )
                             )
            in_channels = out_channels

        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # Calculate the size of the flattened output
        # with torch.no_grad():
        #     x = torch.zeros(1, *input_shape)
        #     for layer in self.layers:
        #         x = layer(x)
        #     num_flat_features = x.view(1, -1).size(1)

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], projection_dim, bias=False),
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # Apply the convolutional layers
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)

        # Flatten the output
        x = torch.reshape(x, [x.shape[0], -1])
        # print(x.shape)

        # Apply the fully connected layers
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        # x    = self.proj_layers(feat)
        # print(x.shape)
        return x

    def get_cnn_feat_out(self, x):
    	# Apply the convolutional layers
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)

        # Flatten the output
        x = x.view(x.shape[0], -1)
        return x

    def get_num_feat(self, input_shape):
    	with torch.no_grad():
    		x = torch.zeros(1, *input_shape).to(config.device)
	    	for layer in self.layers:
	    		x = layer(x)

    	return x.view(1, -1).size(1)
