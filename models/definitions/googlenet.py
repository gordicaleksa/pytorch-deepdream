from collections import namedtuple


import torch
from torchvision import models


from utils.constants import SupportedPretrainedWeights


class GoogLeNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            googlenet = models.googlenet(pretrained=True, progress=show_progress).eval()
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['inception3b', 'inception4c', 'inception4d', 'inception4e']

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
        self.inception4d = googlenet.inception4d
        self.inception4e = googlenet.inception4e

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # todo: not sure why they are using this additional processing - made an issue
    #  https://discuss.pytorch.org/t/why-does-googlenet-additionally-process-input-via-transform-input/88865
    def transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self.transform_input(x)
        # N x 3 x 224 x 224
        x = self.conv1(x)
        conv1 = x
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        mp1 = x
        # N x 64 x 56 x 56
        x = self.conv2(x)
        conv2 = x
        # N x 64 x 56 x 56
        x = self.conv3(x)
        conv3 = x
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        mp2 = x

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        inception3a = x
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        inception3b = x
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        mp3 = x

        # N x 480 x 14 x 14
        x = self.inception4a(x)
        inception4a = x
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        inception4b = x
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        inception4c = x
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        inception4d = x
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        inception4e = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("GoogLeNetOutputs", self.layer_names)
        out = net_outputs(inception3b, inception4c, inception4d, inception4e)
        return out