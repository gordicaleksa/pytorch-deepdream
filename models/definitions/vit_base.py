import os
from collections import namedtuple

import torch
from torchvision import models
import open_clip

from utils.constants import *


class ViT_base(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, model_name = "ViT-B-16", pretrained_weights = SupportedPretrainedWeights.IMAGENET.name, requires_grad=False, show_progress=False):
        super().__init__()

        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            if model_name == 'ViT-B-16':
                vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT, progress=show_progress).eval()

            if model_name == 'ViT-B-32':
                vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT, progress=show_progress).eval()

            self.conv1 = vit.conv_proj
            self.positional_embedding = vit.encoder.pos_embedding
            self.ln_pre = torch.nn.Identity()
            self.layers = vit.encoder.layers
            self.class_token = vit.class_token

        elif pretrained_weights.startswith("CLIP"):
            pretrained_weights = pretrained_weights[5:].lower()
            vit = open_clip.create_model(
                model_name,
                pretrained=pretrained_weights,
                require_pretrained=True
            ).visual.eval()
                
            self.conv1 = vit.conv1
            self.positional_embedding = vit.positional_embedding
            self.ln_pre = vit.ln_pre
            self.class_token = vit.class_embedding
            self.layers = vit.transformer.resblocks

        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} {model_name} model.')

        self.layer_names = [
            'layer0',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'layer5',
            'layer6',
            'layer7',
            'layer8',
            'layer9',
            'layer10',
            'layer11'
            ]

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        activations = []
        
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        for index, layer in enumerate(self.layers):
          x = layer(x)
          activations.append(x)

        # Feel free to experiment with different layers.
        vit_base_outputs = namedtuple("ViT_baseOutputs", self.layer_names)
        out = vit_base_outputs(*activations)
        return out