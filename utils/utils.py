import os
import math
import numbers


import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to('cuda')
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to('cuda')
KERNEL_SIZE = 9


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device, batch_size=1):
    img = load_image(img_path, target_shape=target_shape)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)])
    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)

    return img


def preprocess_img(img):
    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1
    return img


def post_process_image(dump_img, channel_last=False):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    if channel_last:
        dump_img = np.moveaxis(dump_img, 2, 0)

    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)

    return dump_img


def save_and_maybe_display_image(dump_img, should_display=True, channel_last=False, name='test.jpg'):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    dump_img = post_process_image(dump_img, channel_last=channel_last)
    cv.imwrite(name, dump_img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...

    if should_display:
        plt.imshow(dump_img)
        plt.show()


def pytorch_input_adapter(img, device):
    tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    tensor.requires_grad = True
    return tensor


def pytorch_output_adapter(img):
    return np.moveaxis(img.to('cpu').detach().numpy()[0], 0, 2)


# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/48097478#48097478
def create_image_pyramid(img, num_octaves, octave_scale):
    img_pyramid = [img]
    for i in range(num_octaves-1):  # img_pyramid will have "num_octaves" images
        img_pyramid.append(cv.resize(img_pyramid[-1], (0, 0), fx=octave_scale, fy=octave_scale))
    return img_pyramid


def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = True
        return rolled


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, channels, kernel_size, sigma):
        super().__init__()
        dim = 2

        self.pad = int(kernel_size / 2)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        sigmas = [0.5 * sigma, 1.0 * sigma, 2.0 * sigma]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        kernels = []
        for s in sigmas:
            sigma = [s, s]
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                          torch.exp(-((mgrid - mean) / std) ** 2 / 2)
                kernels.append(kernel)

        prepared_kernels = []
        for kernel in kernels:
            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.size())
            kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
            kernel = kernel.to('cuda')
            prepared_kernels.append(kernel)

        self.register_buffer('weight1', prepared_kernels[0])
        self.register_buffer('weight2', prepared_kernels[1])
        self.register_buffer('weight3', prepared_kernels[2])
        self.groups = channels
        self.conv = F.conv2d

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        grad1 = self.conv(input, weight=self.weight1, groups=self.groups)
        grad2 = self.conv(input, weight=self.weight2, groups=self.groups)
        grad3 = self.conv(input, weight=self.weight3, groups=self.groups)
        return grad1 + grad2 + grad3