from collections import namedtuple
import torch
from torchvision import models


from utils.constants import SupportedPretrainedWeights


class AlexNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET:
            alexnet = models.alexnet(pretrained=True, progress=show_progress).eval()
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

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