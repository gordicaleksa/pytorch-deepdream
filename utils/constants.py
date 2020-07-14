import enum


import numpy as np
import torch


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to('cuda')
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to('cuda')
KERNEL_SIZE = 9


class SupportedTransforms(enum.Enum):
    ZOOM = 0
    ROTATE = 1
    SPIRAL = 2


class SupportedModels(enum.Enum):
    VGG16 = 0
    GOOGLENET = 1
    RESNET50 = 2
    ALEXNET = 3


class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1