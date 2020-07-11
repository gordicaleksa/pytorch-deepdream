from collections import namedtuple
import torch
from torchvision import models


class GoogLeNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        # Keeping eval() mode only for consistency - it only affects BatchNorm and Dropout both of which we won't use
        googlenet = models.googlenet(pretrained=True, progress=show_progress).eval()
        self.layer_names = ['inception3b', 'inception4c']

        self.conv1 = googlenet.conv1
        self.maxpool1 = googlenet.maxpool1
        self.conv2 = googlenet.conv2
        self.conv3 = googlenet.conv3
        self.maxpool2 = googlenet.maxpool2
        self.inception3a = googlenet.inception3a
        self.inception3b = googlenet.inception3b
        self.maxpool3 = googlenet.maxpool3
        self.inception4a = googlenet.inception4a
        self.inception4b = googlenet.inception4b
        self.inception4c = googlenet.inception4c

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        inception3b = x
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        inception4c = x

        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(inception3b, inception4c)
        return out