import os
import math
import numbers

import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import scipy.ndimage as nd


from models.definitions.vggs import Vgg16, Vgg16Experimental
from models.definitions.googlenet import GoogLeNet
from models.definitions.resnets import ResNet50
from models.definitions.alexnet import AlexNet
from .constants import *


#
# Image manipulation util functions
#

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


def pre_process_numpy_img(img):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'

    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1  # normalize image
    return img


def post_process_numpy_img(img):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'

    if img.shape[0] == 3:  # if channel-first format move to channel-last (CHW -> HWC)
        img = np.moveaxis(img, 0, 2)

    mean = IMAGENET_MEAN_1.reshape(1, 1, -1)
    std = IMAGENET_STD_1.reshape(1, 1, -1)
    img = (img * std) + mean  # de-normalize
    img = np.clip(img, 0., 1.)  # make sure it's in the [0, 1] range

    return img


def pytorch_input_adapter(img, device):
    # shape = (1, 3, H, W)
    tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    tensor.requires_grad = True
    return tensor


def pytorch_output_adapter(tensor):
    # Push to CPU, detach from the computational graph, convert from (1, 3, H, W) into (H, W, 3)
    return np.moveaxis(tensor.to('cpu').detach().numpy()[0], 0, 2)


def build_image_name(config):
    input_name = 'rand_noise' if config['use_noise'] else config['input_name'].rsplit('.', 1)[0]
    layers = '_'.join(config['layers_to_use'])
    # Looks awful but makes the creation process transparent for other creators
    img_name = f'{input_name}_width_{config["img_width"]}_model_{config["model_name"]}_{config["pretrained_weights"]}_{layers}_pyrsize_{config["pyramid_size"]}_pyrratio_{config["pyramid_ratio"]}_iter_{config["num_gradient_ascent_iterations"]}_lr_{config["lr"]}_shift_{config["spatial_shift_size"]}_smooth_{config["smoothing_coefficient"]}.jpg'
    return img_name


def save_and_maybe_display_image(config, dump_img, name_modifier=None):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    # step1: figure out the dump dir location
    dump_dir = config['dump_dir']
    os.makedirs(dump_dir, exist_ok=True)

    # step2: define the output image name
    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + '.jpg'
    else:
        dump_img_name = build_image_name(config)

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img*255).astype(np.uint8)

    # step3: write image to the file system
    # ::-1 because opencv expects BGR (and not RGB) format...
    dump_path = os.path.join(dump_dir, dump_img_name)
    cv.imwrite(dump_path, dump_img[:, :, ::-1])

    # step4: potentially display/plot the image
    if config['should_display']:
        plt.imshow(dump_img)
        plt.show()

    return dump_path


def linear_blend(img1, img2, alpha=0.5):
    return img1 + alpha * (img2 - img1)


#
# End of image manipulation util functions
#


def fetch_and_prepare_model(model_type, pretrained_weights, device):
    if model_type == SupportedModels.VGG16.name:
        model = Vgg16(pretrained_weights, requires_grad=False, show_progress=True).to(device)
    elif model_type == SupportedModels.VGG16_EXPERIMENTAL.name:
        model = Vgg16Experimental(pretrained_weights, requires_grad=False, show_progress=True).to(device)
    elif model_type == SupportedModels.GOOGLENET.name:
        model = GoogLeNet(pretrained_weights, requires_grad=False, show_progress=True).to(device)
    elif model_type == SupportedModels.RESNET50.name:
        model = ResNet50(pretrained_weights, requires_grad=False, show_progress=True).to(device)
    elif model_type == SupportedModels.ALEXNET.name:
        model = AlexNet(pretrained_weights, requires_grad=False, show_progress=True).to(device)
    else:
        raise Exception('Model not yet supported.')
    return model


# Didn't want to expose these to the outer API - too much clutter, feel free to tweak params here
def transform_frame(config, frame):
    h, w = frame.shape[:2]
    ref_fps = 30  # referent fps, the transformation settings are calibrated for this one

    if config['frame_transform'].lower() == TRANSFORMS.ZOOM.name.lower():
        scale = 1.04 * (ref_fps / config['fps'])  # Use this param to (un)zoom
        rotation_matrix = cv.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        frame = cv.warpAffine(frame, rotation_matrix, (w, h))

    elif config['frame_transform'].lower() == TRANSFORMS.ZOOM_ROTATE.name.lower():
        # Arbitrary heuristic keep the degree at 3 degrees/second and scale 1.04/second
        deg = 1.5 * (ref_fps / config['fps'])  # Adjust rotation speed (in [deg/frame])
        scale = 1.04 * (ref_fps / config['fps'])  # Use this to (un)zoom while rotating around image center
        rotation_matrix = cv.getRotationMatrix2D((w / 2, h / 2), deg, scale)
        frame = cv.warpAffine(frame, rotation_matrix, (w, h))

    elif config['frame_transform'].lower() == TRANSFORMS.TRANSLATE.name.lower():
        tx, ty = [2 * (ref_fps / config['fps']), 2 * (ref_fps / config['fps'])]
        translation_matrix = np.asarray([[1., 0., tx], [0., 1., ty]])
        frame = cv.warpAffine(frame, translation_matrix, (w, h))

    else:
        raise Exception('Transformation not yet supported.')

    return frame


def get_new_shape(config, base_shape, pyramid_level):
    SHAPE_MARGIN = 10
    pyramid_ratio = config['pyramid_ratio']
    pyramid_size = config['pyramid_size']
    exponent = pyramid_level - pyramid_size + 1
    new_shape = np.round(np.float32(base_shape)*(pyramid_ratio**exponent)).astype(np.int32)

    if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
        print(f'Pyramid size {config["pyramid_size"]} with pyramid ratio {config["pyramid_ratio"]} gives too small pyramid levels with size={new_shape}')
        print(f'Please change parameters.')
        exit(0)

    return new_shape


def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = True
        return rolled


class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(DEVICE)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3


# Not used atm.
def create_image_pyramid(img, num_octaves, octave_scale):
    img_pyramid = [img]
    for i in range(num_octaves-1):  # img_pyramid will have "num_octaves" images
        img_pyramid.append(cv.resize(img_pyramid[-1], (0, 0), fx=1./octave_scale, fy=1./octave_scale))
    return img_pyramid


def print_deep_dream_video_header(config):
    print(f'Creating a DeepDream video from {config["input_name"]}, via {config["model_name"]} model.')
    print(f'Using pretrained weights = {config["pretrained_weights"]}')
    print(f'Using model layers = {config["layers_to_use"]}')
    print(f'Using lending coefficient = {config["blend"]}.')
    print(f'Video output width = {config["img_width"]}')
    print(f'fps = {config["fps"]}')
    print('*' * 50, '\n')


def print_ouroboros_video_header(config):
    print(f'Creating a {config["ouroboros_length"]}-frame Ouroboros video from {config["input_name"]}, via {config["model_name"]} model.')
    print(f'Using {config["frame_transform"]} for the frame transform')
    print(f'Using pretrained weights = {config["pretrained_weights"]}')
    print(f'Using model layers = {config["layers_to_use"]}')
    print(f'Video output width = {config["img_width"]}')
    print(f'fps = {config["fps"]}')
    print('*' * 50, '\n')


def parse_input_file(input):
    # Handle abs/rel paths
    if os.path.exists(input):
        return input
    # If passed only a name and it doesn't exist in the current working dir assume it's in input data dir
    elif os.path.exists(os.path.join(INPUT_DATA_PATH, input)):
        return os.path.join(INPUT_DATA_PATH, input)
    else:
        raise Exception(f'Input path {input} is not valid.')
