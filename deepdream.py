import os
import argparse


import numpy as np
import scipy.ndimage as nd
import torch
from torch.optim import Adam
import cv2 as cv


from models.vggs import Vgg16
from models.googlenet import GoogLeNet
import utils.utils as utils
from utils.utils import LOWER_IMAGE_BOUND, UPPER_IMAGE_BOUND, GaussianSmoothing, KERNEL_SIZE

# todo: experiment with different models (GoogLeNet, pytorch models trained on MIT Places?)
# todo: experiment with different single/multiple layers
# todo: add guide

# todo: add random init image support


def gradient_ascent(backbone_network, img, lr, cnt):
    out = backbone_network(img)
    layer = out.relu4_3
    # loss = torch.norm(torch.flatten(layer), p=2)
    loss = torch.nn.MSELoss(reduction='sum')(layer, torch.zeros_like(layer)) / 2
    # layer.backward(layer)

    loss.backward()
    # todo: [1] other models trained on non-ImageNet datasets
    #  if I still don't get reasonable video stream

    grad = img.grad.data

    sigma = ((cnt + 1) / 10) * 2.0 + .5
    smooth_grad = GaussianSmoothing(3, KERNEL_SIZE, sigma)(grad)

    # todo: consider using std for grad normalization and not mean
    g_mean = torch.mean(torch.abs(smooth_grad))
    img.data += lr * (smooth_grad / g_mean)
    img.grad.data.zero_()

    img.data = torch.max(torch.min(img, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def gradient_ascent_adam(backbone_network, img):
    optimizer = Adam((img,), lr=0.09)

    out = backbone_network(img)
    layer = out.inception4c
    loss = -torch.nn.MSELoss(reduction='sum')(layer, torch.zeros_like(layer)) / 2
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    img.data = torch.max(torch.min(img, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)  # https://stackoverflow.com/questions/54738045/column-dependent-bounds-in-torch-clamp


# Contains the gist of DeepDream algorithm - takes 15 minutes to write down
# no spatial jitter, no octaves, no clipping policy, no advanced gradient normalization (std)
def deep_dream_simple(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = utils.prepare_img(img_path, target_shape=500, device=device)
    img.requires_grad = True
    backbone_network = Vgg16(requires_grad=False).to(device)

    n_iter = 2
    lr = 0.2

    for iter in range(n_iter):
        gradient_ascent(backbone_network, img, lr)

    img = img.to('cpu').detach().numpy()[0]
    utils.save_and_maybe_display_image(img)


def deep_dream_static_image(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Vgg16(requires_grad=False, show_progress=True).to(device)
    base_img = utils.preprocess_img(img)

    # todo: experiment with these
    pyramid_size = 2
    pyramid_ratio = 1./1.4
    n_iter = 10
    lr = 0.09
    jitter = 100

    # contains pyramid_size copies of the very same image with different resolutions
    img_pyramid = utils.create_image_pyramid(base_img, pyramid_size, pyramid_ratio)

    detail = np.zeros_like(img_pyramid[-1])  # allocate image for network-produced details

    best_img = []
    # going from smaller to bigger resolution
    for octave, octave_base in enumerate(reversed(img_pyramid)):
        h, w = octave_base.shape[:2]
        if octave > 0:  # we can avoid this special case
            # upscale details from the previous octave
            h1, w1 = detail.shape[:2]
            detail = cv.resize(detail, (w, h))
        input_img = octave_base + detail
        input_tensor = utils.pytorch_input_adapter(input_img, device)
        for i in range(n_iter):
            h_shift, w_shift = np.random.randint(-jitter, jitter + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            # print(tmp.requires_grad, input_tensor.requires_grad)
            gradient_ascent(net, input_tensor, lr, i)
            # gradient_ascent_adam(net, input_tensor)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
            # visualization
            # current_img = pytorch_output_adapter(input_tensor)
            # print(current_img.shape)
            # vis = post_process_image(current_img, channel_last=True)
            # plt.imshow(vis); plt.show()

        # todo: consider just rescaling without doing subtraction
        detail = utils.pytorch_output_adapter(input_tensor) - octave_base

        current_img = utils.pytorch_output_adapter(input_tensor)
        best_img = current_img
        # save_and_maybe_display_image(current_img, channel_last=True)
    return best_img


def deep_dream_video(config):
    img_path = os.path.join(config['input_images_path'], config['input_img_name'])
    frame = utils.load_image(img_path, target_shape=config['img_width'])

    s = 0.05  # scale coefficient
    for frame_id in range(config['video_length']):
        print(f'Dream iteration {frame_id+1}.')
        frame = deep_dream_static_image(frame)
        h, w = frame.shape[:2]
        utils.save_and_maybe_display_image(frame, channel_last=True, should_display=False, name=os.path.join('data/out-videos/video', str(frame_id) + '.jpg'))
        # todo: make it more declarative and not imperative, rotate, zoom, etc. nice API
        frame = nd.affine_transform(frame, np.asarray([1 - s, 1 - s, 1]), [h * s / 2, w * s / 2, 0.0], order=1)


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    input_images_path = os.path.join(os.path.dirname(__file__), 'data', 'input-images')

    #
    # Modifiable args - feel free to play with these (only a small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_video", type=bool, help="Create DeepDream video - default is DeepDream image", default=True)
    parser.add_argument("--video_length", type=int, help="Number of video frames to produce", default=30)
    parser.add_argument("--input_img_name", type=str, help="Input image name that will be used for dreaming", default='figures.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=500)
    args = parser.parse_args()

    # Wrapping configuration into a dictionary - keeps things clean
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['input_images_path'] = input_images_path

    # DeepDream algorithm
    if config['is_video']:
        deep_dream_video(config)
    else:
        deep_dream_static_image("dummy")


