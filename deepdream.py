import os
import argparse


import numpy as np
import torch
from torch.optim import Adam
import cv2 as cv
import shutil


from models.vggs import Vgg16
import utils.utils as utils
from utils.utils import LOWER_IMAGE_BOUND, UPPER_IMAGE_BOUND, GaussianSmoothing, KERNEL_SIZE, SUPPORTED_TRANSFORMS, SUPPORTED_MODELS
import utils.video_utils as video_utils


# todo: experiment with different models (GoogLeNet, pytorch models trained on MIT Places?) can I use caffe models?
# todo: add guide


# Not used atm
def gradient_ascent_adam(backbone_network, img):
    optimizer = Adam((img,), lr=0.09)

    out = backbone_network(img)
    layer = out.inception4c
    loss = -torch.nn.MSELoss(reduction='sum')(layer, torch.zeros_like(layer)) / 2
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    img.data = torch.max(torch.min(img, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)  # https://stackoverflow.com/questions/54738045/column-dependent-bounds-in-torch-clamp


# layer_activation.backward(layer) <- original implementation <=> with MSE / 2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    out = model(input_tensor)
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]
    losses = []
    for layer_activation in activations:
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation)) # torch.norm(torch.flatten(layer_activation), p=2)
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()
    # todo: [2] other models trained on non-ImageNet datasets
    #  if I still don't get reasonable video stream

    grad = input_tensor.grad.data

    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + .5
    smooth_grad = GaussianSmoothing(3, KERNEL_SIZE, sigma)(grad)

    g_norm = torch.std(smooth_grad)  # g_norm = torch.mean(torch.abs(smooth_grad))
    input_tensor.data += config['lr'] * (smooth_grad / g_norm)
    input_tensor.grad.data.zero_()

    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    model = utils.fetch_and_prepare_model(config['model'], device)
    layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layer_to_use']]

    if img is None:  # in case the image wasn't specified load either image or start from noise
        img_path = os.path.join(config['inputs_path'], config['input'])
        img = utils.load_image(img_path, target_shape=config['img_width'])  # load numpy, [0, 1], channel-last, RGB image
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.preprocess_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width

    # Note: simple rescaling the whole result and not only details (see original implementation) gave me better results
    # going from smaller to bigger resolution
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, device)

        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)

            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)  # gradient_ascent_adam(model, input_tensor)

            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)

        img = utils.pytorch_output_adapter(input_tensor)

    return img


# todo: checkout feed output into input DeepDream video whether they reference how they approached this
def deep_dream_video_ouroboros(config):
    img_path = os.path.join(config['inputs_path'], config['input'])
    # load numpy, [0, 1], channel-last, RGB image, None will cause it to start from the uniform noise [0, 1] image
    frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])

    for frame_id in range(config['video_length']):
        print(f'Dream iteration {frame_id+1}.')
        frame = deep_dream_static_image(config, frame)
        utils.save_and_maybe_display_image(config, frame, should_display=config['should_display'], name_modifier=frame_id)
        frame = utils.transform_frame(config, frame)  # transform frame e.g. central zoom, spiral, etc.

    # todo: add video creation function from frames
    video_utils.create_video_from_intermediate_results(config)


# todo: create video by applying deep_dream_static_image per video frame and packing those into video
# todo: add blend support
# todo: add optical flow support
def deep_dream_video(config):
    video_path = os.path.join(config['inputs_path'], config['input'])
    tmp_dump_dir = os.path.join(config['out_videos_path'], 'tmp')
    config['dump_dir'] = tmp_dump_dir
    os.makedirs(tmp_dump_dir, exist_ok=True)

    video_utils.dump_frames(video_path, tmp_dump_dir)

    for frame_id, frame_name in enumerate(os.listdir(tmp_dump_dir)):
        frame_path = os.path.join(tmp_dump_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config['img_width'])
        dreamed_frame = deep_dream_static_image(config, frame)
        utils.save_and_maybe_display_image(config, dreamed_frame, should_display=config['should_display'], name_modifier=frame_id)

    # todo: [2] make it generic enough to be able to called from wherever
    video_path = video_utils.create_video_from_intermediate_results(config)

    shutil.move(video_path, config['out_videos_path'])  # move final video to out videos path

    shutil.rmtree(tmp_dump_dir)  # remove tmp files
    print(f'Deleted tmp frame dump directory {tmp_dump_dir}.')


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    inputs_path = os.path.join(os.path.dirname(__file__), 'data', 'input-images')
    out_images_path = os.path.join(os.path.dirname(__file__), 'data', 'out-images')
    out_videos_path = os.path.join(os.path.dirname(__file__), 'data', 'out-videos')

    #
    # Modifiable args - feel free to play with these (only a small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_video", type=bool, help="Create DeepDream video - default is DeepDream image", default=True)
    parser.add_argument("--video_length", type=int, help="Number of video frames to produce", default=100)
    parser.add_argument("--input", type=str, help="Input image/video name that will be used for dreaming", default='video.mp4')
    parser.add_argument("--use_noise", type=bool, help="Use noise as a starting point instead of input image", default=False)
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, help="Neural network (model) to use for dreaming", default=SUPPORTED_MODELS[0])
    parser.add_argument("--layer_to_use", type=str, help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])
    parser.add_argument("--frame_transform", type=str, choices=SUPPORTED_TRANSFORMS,
                        help="Transform used to transform the output frame and feed it back to the network input", default=SUPPORTED_TRANSFORMS[0])

    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.3)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.2)
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)

    parser.add_argument("--should_display", type=bool, help="Display intermediate dreaming results", default=False)
    args = parser.parse_args()

    # Wrapping configuration into a dictionary - keeping things clean
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['inputs_path'] = inputs_path
    config['out_images_path'] = out_images_path
    config['out_videos_path'] = out_videos_path
    config['dump_dir'] = config['out_videos_path'] if config['is_video'] else config['out_images_path']

    # DeepDream algorithm
    if config['input'].endswith('.mp4'):  # only support mp4 atm
        deep_dream_video(config)
    elif config['is_video']:
        deep_dream_video_ouroboros(config)
    else:
        deep_dream_static_image(config, img=None)  # img will be loaded inside of deep_dream_static_image


