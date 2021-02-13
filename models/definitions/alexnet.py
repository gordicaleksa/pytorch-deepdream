import os
from collections import namedtuple


import torch
from torchvision import models
from torch.hub import download_url_to_file


from utils.constants import *


class AlexNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            alexnet = models.alexnet(pretrained=True, progress=show_progress).eval()
        else:
            alexnet = models.alexnet(pretrained=False, progress=show_progress).eval()

            binary_name = 'alexnet_places365.pth.tar'
            alexnet_places365_binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(alexnet_places365_binary_path):
                state_dict = torch.load(alexnet_places365_binary_path)['state_dict']
            else:
                binary_url = r'http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                download_url_to_file(binary_url, alexnet_places365_binary_path)
                print('Done downloading.')
                state_dict = torch.load(alexnet_places365_binary_path)['state_dict']

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key.replace('.module', '')
                new_state_dict[new_key] = state_dict[old_key]

            alexnet.classifier[-1] = torch.nn.Linear(alexnet.classifier[-1].in_features, 365)
            alexnet.load_state_dict(new_state_dict, strict=True)

        alexnet_pretrained_features = alexnet.features
        self.layer_names = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1 = x
        x = self.slice2(x)
        relu2 = x
        x = self.slice3(x)
        relu3 = x
        x = self.slice4(x)
        relu4 = x
        x = self.slice5(x)
        relu5 = x

        alexnet_outputs = namedtuple("AlexNetOutputs", self.layer_names)
        out = alexnet_outputs(relu1, relu2, relu3, relu4, relu5)
        return out