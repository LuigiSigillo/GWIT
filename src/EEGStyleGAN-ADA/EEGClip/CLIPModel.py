# Reference for Projection Head
# https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-with-PyTorch-Lightning--VmlldzoyMzg4Njk1

import config
import torch
from torch import nn

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
        # self.gelu = nn.GELU()
        # self.fc = nn.Linear(projection_dim, projection_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        # x = self.gelu(projected)
        # x = self.fc(x)
        # x = self.dropout(x)
        # x += projected
        # return self.layer_norm(x)
        return projected


class CLIPModel(torch.nn.Module):
    def __init__(self, text_encoder, image_encoder, embedding_dim, projection_dim, dropout=0.01):
        super(CLIPModel, self).__init__()
        self.text_encoder     = text_encoder
        self.image_encoder    = image_encoder
        # self.text_projection  = ProjectionHead(embedding_dim, projection_dim, dropout).to(config.device)
        # self.image_projection = ProjectionHead(embedding_dim, projection_dim, dropout).to(config.device)

    
    def forward(self, texts, images):
        text_feat  = self.text_encoder(texts)
        image_feat = self.image_encoder(images)
        # text_embed = self.text_projection( text_feat )
        # image_embed = self.image_projection( image_feat )
        text_embed  = torch.nn.functional.normalize( text_feat, dim=-1 )
        image_embed = torch.nn.functional.normalize( image_feat, dim=-1 )

        return text_embed, image_embed, text_feat, image_feat
