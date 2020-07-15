import os
import argparse
import shutil


import numpy as np
import torch
import cv2 as cv


import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils


# todo: can I adapt original caffe models/weights to PyTorch?
# todo: add guide

# todo: [1] README
# todo: [2] figure out which data to checkin
# todo: [3] Refactor

# comment: Places 365 models are not giving good results
# - preprocessing is the same as for ImageNet, so that's out of the way


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

    # Process image gradients
    grad = input_tensor.grad.data

    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + .5
    smooth_grad = utils.GaussianSmoothing(3, KERNEL_SIZE, sigma)(grad)

    g_norm = torch.std(smooth_grad)  # g_norm = torch.mean(torch.abs(smooth_grad))

    # Update image using the calculated gradients
    input_tensor.data += config['lr'] * (smooth_grad / g_norm)

    # Clear gradients and clamp the data (otherwise it would explode to +- Inf)
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    model = utils.fetch_and_prepare_model(config['model'], config['pretrained_weights'], device)
    layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]

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

    return utils.post_process_numpy_image(img)


def deep_dream_video_ouroboros(config):
    img_path = os.path.join(config['inputs_path'], config['input'])
    # load numpy, [0, 1], channel-last, RGB image, None will cause it to start from the uniform noise [0, 1] image
    frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])

    for frame_id in range(config['video_length']):
        print(f'Dream iteration {frame_id+1}.')
        frame = deep_dream_static_image(config, frame)
        utils.save_and_maybe_display_image(config, frame, should_display=config['should_display'], name_modifier=frame_id)
        frame = utils.transform_frame(config, frame)  # transform frame e.g. central zoom, spiral, etc.

    video_utils.create_video_from_intermediate_results(config)


# todo: add optical flow support
def deep_dream_video(config):
    video_path = os.path.join(config['inputs_path'], config['input'])
    tmp_input_dir = os.path.join(config['out_videos_path'], 'tmp_input')
    tmp_output_dir = os.path.join(config['out_videos_path'], 'tmp_out4')
    config['dump_dir'] = tmp_output_dir
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)

    metadata = video_utils.dump_frames(video_path, tmp_input_dir)

    last_img = None
    for frame_id, frame_name in enumerate(os.listdir(tmp_input_dir)):
        print(f'Processing frame {frame_id}')
        frame_path = os.path.join(tmp_input_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config['img_width'])
        if config['blend'] is not None and last_img is not None:
            frame = utils.linear_blend(last_img, frame, config['blend'])

        dreamed_frame = deep_dream_static_image(config, frame)
        last_img = dreamed_frame
        utils.save_and_maybe_display_image(config, dreamed_frame, should_display=config['should_display'], name_modifier=frame_id)

    video_utils.create_video_from_intermediate_results(config, metadata)

    shutil.rmtree(tmp_input_dir)  # remove tmp files
    # shutil.rmtree(tmp_output_dir)  # remove tmp files
    print(f'Deleted tmp frame dump directory {tmp_input_dir}.')


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    inputs_path = os.path.join(os.path.dirname(__file__), 'data', 'input')
    out_images_path = os.path.join(os.path.dirname(__file__), 'data', 'out-images')
    out_videos_path = os.path.join(os.path.dirname(__file__), 'data', 'out-videos')

    #
    # Modifiable args - feel free to play with these (only a small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_video", type=bool, help="Create DeepDream video - default is DeepDream image", default=False)
    parser.add_argument("--video_length", type=int, help="Number of video frames to produce", default=100)
    parser.add_argument("--input", type=str, help="Input image/video name that will be used for dreaming", default='figures.jpg')
    parser.add_argument("--use_noise", type=bool, help="Use noise as a starting point instead of input image", default=False)
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--model", choices=SupportedModels, help="Neural network (model) to use for dreaming", default=SupportedModels.VGG16)
    parser.add_argument("--pretrained_weights", choices=SupportedPretrainedWeights, help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.IMAGENET)
    parser.add_argument("--layers_to_use", type=str, help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])
    parser.add_argument("--frame_transform", choices=SupportedTransforms,
                        help="Transform used to transform the output frame and feed it back to the network input", default=SupportedTransforms.ZOOM)

    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.3)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)

    parser.add_argument("--blend", type=float, help="Blend coefficient for video creation", default=0.85)

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

    # DeepDream algorithm in 3 flavours: static image, video and ouroboros (feeding net output to it's input)
    if config['input'].endswith('.mp4'):  # only support mp4 atm
        deep_dream_video(config)
    elif config['is_video']:
        deep_dream_video_ouroboros(config)
    else:
        img = deep_dream_static_image(config, img=None)  # img will be loaded inside of deep_dream_static_image
        utils.save_and_maybe_display_image(config, img, should_display=False)

