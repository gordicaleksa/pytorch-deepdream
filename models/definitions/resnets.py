import os
from collections import namedtuple

import torch
from torchvision import models
from torch.hub import download_url_to_file
import open_clip

from utils.constants import *


class ResNet(torch.nn.Module):

    def __init__(self, model_name = "RN50", pretrained_weights = SupportedPretrainedWeights.IMAGENET.name, requires_grad=False, show_progress=False):
        super().__init__()

        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            if model_name == 'RN50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=show_progress).eval()

            if model_name == 'RN101':
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT, progress=show_progress).eval()

            if model_name == 'RN152':
                model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=show_progress).eval()

            self.conv1 = model.conv1
            self.conv2 = torch.nn.Identity()
            self.conv3 = torch.nn.Identity()
            self.bn1 = model.bn1
            self.bn2 = torch.nn.Identity()
            self.bn3 = torch.nn.Identity()
            self.relu1 = model.relu
            self.relu2 = torch.nn.Identity()
            self.relu3 = torch.nn.Identity()
            self.pool = model.maxpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

        elif (pretrained_weights == SupportedPretrainedWeights.PLACES_365.name) and (model_name == "RN50"):
            model = models.resnet50(pretrained=False, progress=show_progress).eval()

            binary_name = 'resnet50_places365.pth.tar'
            resnet50_places365_binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(resnet50_places365_binary_path):
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']
            else:
                binary_url = r'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                download_url_to_file(binary_url, resnet50_places365_binary_path)
                print('Done downloading.')
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key[7:]
                new_state_dict[new_key] = state_dict[old_key]

            model.fc = torch.nn.Linear(model.fc.in_features, 365)
            model.load_state_dict(new_state_dict, strict=True)

            self.conv1 = model.conv1
            self.conv2 = torch.nn.Identity()
            self.conv3 = torch.nn.Identity()
            self.bn1 = model.bn1
            self.bn2 = torch.nn.Identity()
            self.bn3 = torch.nn.Identity()
            self.relu1 = model.relu
            self.relu2 = torch.nn.Identity()
            self.relu3 = torch.nn.Identity()
            self.pool = model.maxpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
        

        elif pretrained_weights.startswith("CLIP"):
            pretrained_weights = pretrained_weights[5:].lower()

            model = open_clip.create_model(
                model_name,
                pretrained=pretrained_weights,
                require_pretrained=True
            ).visual.eval()
            
            self.conv1 = model.conv1
            self.conv2 = model.conv2
            self.conv3 = model.conv3
            self.bn1 = model.bn1
            self.bn2 = model.bn2
            self.bn3 = model.bn3
            self.relu1 = model.act1
            self.relu2 = model.act2
            self.relu3 = model.act3
            self.pool = model.avgpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} {model_name} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        resnet_outputs = namedtuple("ResNetOutputs", self.layer_names)
        out = resnet_outputs(layer1, layer2, layer3, layer4)
        return out
    

class ResNet50Experimental(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            resnet50 = models.resnet50(pretrained=True, progress=show_progress).eval()
        elif pretrained_weights == SupportedPretrainedWeights.PLACES_365.name:
            resnet50 = models.resnet50(pretrained=False, progress=show_progress).eval()

            binary_name = 'resnet50_places365.pth.tar'
            resnet50_places365_binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(resnet50_places365_binary_path):
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']
            else:
                binary_url = r'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                download_url_to_file(binary_url, resnet50_places365_binary_path)
                print('Done downloading.')
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key[7:]
                new_state_dict[new_key] = state_dict[old_key]

            resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 365)
            resnet50.load_state_dict(new_state_dict, strict=True)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # 3
        self.layer10 = resnet50.layer1[0]
        self.layer11 = resnet50.layer1[1]
        self.layer12 = resnet50.layer1[2]

        # 4
        self.layer20 = resnet50.layer2[0]
        self.layer21 = resnet50.layer2[1]
        self.layer22 = resnet50.layer2[2]
        self.layer23 = resnet50.layer2[3]

        # 6
        self.layer30 = resnet50.layer3[0]
        self.layer31 = resnet50.layer3[1]
        self.layer32 = resnet50.layer3[2]
        self.layer33 = resnet50.layer3[3]
        self.layer34 = resnet50.layer3[4]
        self.layer35 = resnet50.layer3[5]

        # 3
        self.layer40 = resnet50.layer4[0]
        self.layer41 = resnet50.layer4[1]
        # self.layer42 = resnet50.layer4[2]

        # Go even deeper into ResNet's BottleNeck module for layer 42
        self.layer42_conv1 = resnet50.layer4[2].conv1
        self.layer42_bn1 = resnet50.layer4[2].bn1
        self.layer42_conv2 = resnet50.layer4[2].conv2
        self.layer42_bn2 = resnet50.layer4[2].bn2
        self.layer42_conv3 = resnet50.layer4[2].conv3
        self.layer42_bn3 = resnet50.layer4[2].bn3
        self.layer42_relu = resnet50.layer4[2].relu

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        x = self.layer12(x)
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        x = self.layer22(x)
        layer22 = x
        x = self.layer23(x)
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        x = self.layer32(x)
        layer32 = x
        x = self.layer33(x)
        layer33 = x
        x = self.layer34(x)
        layer34 = x
        x = self.layer35(x)
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        x = self.layer42_conv3(x)
        layer426 = x
        x = self.layer42_bn3(x)
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        out = net_outputs(layer10, layer23, layer35, layer40)
        # layer35 is my favourite
        return out