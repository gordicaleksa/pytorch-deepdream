import enum


import numpy as np
import torch

from .device import device


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
KERNEL_SIZE = 9  # "magic number" picked this one as it just works well


class SupportedTransforms(enum.Enum):
    ZOOM = 0
    ZOOM_ROTATE = 1
    TRANSLATE = 2


class SupportedModels(enum.Enum):
    VGG16 = 0
    VGG16_EXPERIMENTAL = 1
    GOOGLENET = 2
    RESNET50 = 3
    ALEXNET = 4


class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1


SUPPORTED_VIDEO_FORMATS = ['.mp4']
