from collections import namedtuple, defaultdict

import torch
import open_clip

from utils.constants import *


class OpenCLIP(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, model_name="RN50", pretrained_weights=None, requires_grad=False):
        super().__init__()

        if pretrained_weights:
            pretrained = pretrained_weights[5:].lower()
        else:
            pretrained = None
            
        if (pretrained is None) or (pretrained in open_clip.list_pretrained_tags_by_model(model_name)):
            self.model = open_clip.create_model(model_name, pretrained=pretrained, require_pretrained=True).eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} {model_name} model.')
        
        self.layer_names = ["logits_per_image"]

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
          for param in self.parameters():
            param.requires_grad = False

        # Store the text features in order to compute them once while deepdreaming
        self.stored_text_features = defaultdict(lambda: None)

    def forward(self, x):
        img, text = x
        img = img

        if self.stored_text_features[text] is None:
            text_features = self.tokenizer(text).to(DEVICE)
            text_features = self.model.encode_text(text_features)
            self.stored_text_features[text] = text_features
        else:
            text_features = self.stored_text_features[text]

        image_features = self.model.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        out = (100.0 * image_features @ text_features.T)
        
        # Feel free to experiment with different layers.
        openclip_outputs = namedtuple("OpenCLIPOutputs", self.layer_names)
        out = openclip_outputs(out)
        return out






