import os
from collections import namedtuple

import torch
from torchvision import models
import open_clip

from utils.constants import *

class ConvNeXt_large(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, pretrained_weights = SupportedPretrainedWeights.IMAGENET.name, requires_grad=False, show_progress=False):
        super().__init__()

        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name: 
            convnext = models.convnext_large(weights=models.convnext.ConvNeXt_Large_Weights.DEFAULT, progress=show_progress).eval()
            
            self.layer0 = convnext.features[0] 
            self.layer1 = convnext.features[1] 
            self.layer2 = convnext.features[2:4] 
            self.layer3 = convnext.features[4:6] 
            self.layer4 = convnext.features[6:8] 

        elif pretrained_weights.startswith("CLIP"):
            pretrained_weights = pretrained_weights[5:].lower()
            
            if pretrained_weights in open_clip.list_pretrained_tags_by_model("convnext_large_d"):
                convnext = open_clip.create_model(
                    "convnext_large_d", 
                    pretrained=pretrained_weights, 
                    require_pretrained=True
                ).visual.eval()
    
                self.layer0 = convnext.trunk.stem
                self.layer1 = convnext.trunk.stages[0] 
                self.layer2 = convnext.trunk.stages[1] 
                self.layer3 = convnext.trunk.stages[2] 
                self.layer4 = convnext.trunk.stages[3]

            elif pretrained_weights in open_clip.list_pretrained_tags_by_model("convnext_large_d_320"):
                convnext = open_clip.create_model(
                    "convnext_large_d_320", 
                    pretrained=pretrained_weights, 
                    require_pretrained=True
                ).visual.eval()
    
                self.layer0 = convnext.trunk.stem
                self.layer1 = convnext.trunk.stages[0] 
                self.layer2 = convnext.trunk.stages[1] 
                self.layer3 = convnext.trunk.stages[2] 
                self.layer4 = convnext.trunk.stages[3]

        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x

        # Feel free to experiment with different layers.
        convnext_outputs = namedtuple("ConvNeXtOutputs", self.layer_names)
        out = convnext_outputs(layer1, layer2, layer3, layer4)
        return out