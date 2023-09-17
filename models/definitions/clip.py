import torch
import cv2 as cv
import clip
from collections import namedtuple

from utils.constants import *


class CLIP(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, model_name="ViT-B/16", pretrained_weights = SupportedPretrainedWeights.CLIP_OPENAI.name, requires_grad=False):
        super().__init__()
        
        if (pretrained_weights is None) or (pretrained_weights == SupportedPretrainedWeights.CLIP_OPENAI.name):
            self.model = clip.load(model_name, device=DEVICE)[0].eval()
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')
        
        self.layer_names = ["logits_per_image"]

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
          for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        img, text = x
        img = img
        text = clip.tokenize(text).to(DEVICE)

        logits_per_image, logits_per_text = self.model(img, text)
        out = logits_per_image
        
        # Feel free to experiment with different layers.
        clip_outputs = namedtuple("CLIPOutputs", self.layer_names)
        out = clip_outputs(out)
        return out

