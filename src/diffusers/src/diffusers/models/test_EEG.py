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

embedding = test()
print(embedding.shape)
assert embedding.shape == torch.Size([4, 320, 64, 64])

# import torch
# import torch.nn as nn

# # Initial tensor of shape (4, 128, 512)
# x = torch.randn(4, 128, 512)

# # Define the 1D convolutional layers
# conv1d_layers = nn.Sequential(
#     nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),  # (4, 256, 512)
#     nn.ReLU(),
#     nn.Conv1d(256, 320, kernel_size=3, stride=2, padding=1),  # (4, 320, 256)
#     nn.ReLU(),
#     nn.Conv1d(320, 320, kernel_size=3, stride=2, padding=1),  # (4, 320, 128)
#     nn.ReLU(),
#     nn.Conv1d(320, 320, kernel_size=3, stride=2, padding=1),  # (4, 320, 64)
#     nn.ReLU()
# )

# # Apply the 1D convolutional layers
# x = conv1d_layers(x)

# # Reshape to (4, 320, 64, 1)
# x = x.view(4, 320, 64, 1)

# # Pad to (4, 320, 64, 64)
# padding = (0, 63, 0, 0)  # Pad the last dimension to 64
# x = nn.functional.pad(x, padding, mode='constant', value=0)

# # Check the final shape
# print(x.shape)  # Should be torch.Size([4, 320, 64, 64])