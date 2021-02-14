"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import os
import argparse
import shutil
import time


import numpy as np
import torch
import cv2 as cv


import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils


# loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor)

    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config['lr'] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    model = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except Exception as e:  # making sure you set the correct layer name for this specific model
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')
        return

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = utils.parse_input_file(config['input'])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = utils.load_image(img_path, target_shape=config['img_width'])
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)

        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)

            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)

            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)

        img = utils.pytorch_output_adapter(input_tensor)

    return utils.post_process_numpy_img(img)


def deep_dream_video_ouroboros(config):
    """
    Feeds the output dreamed image back to the input and repeat

    Name etymology for nerds: https://en.wikipedia.org/wiki/Ouroboros

    """
    ts = time.time()
    assert any([config['input_name'].lower().endswith(img_ext) for img_ext in SUPPORTED_IMAGE_FORMATS]), \
        f'Expected an image, but got {config["input_name"]}. Supported image formats {SUPPORTED_IMAGE_FORMATS}.'

    utils.print_ouroboros_video_header(config)  # print some ouroboros-related metadata to the console

    img_path = utils.parse_input_file(config['input'])
    # load numpy, [0, 1] range, channel-last, RGB image
    # use_noise and consequently None value, will cause it to initialize the frame with uniform, [0, 1] range, noise
    frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])

    for frame_id in range(config['ouroboros_length']):
        print(f'Ouroboros iteration {frame_id+1}.')
        # Step 1: apply DeepDream and feed the last iteration's output to the input
        frame = deep_dream_static_image(config, frame)
        dump_path = utils.save_and_maybe_display_image(config, frame, name_modifier=frame_id)
        print(f'Saved ouroboros frame to: {os.path.relpath(dump_path)}\n')

        # Step 2: transform frame e.g. central zoom, spiral, etc.
        # Note: this part makes amplifies the psychodelic-like appearance
        frame = utils.transform_frame(config, frame)

    video_utils.create_video_from_intermediate_results(config)
    print(f'time elapsed = {time.time()-ts} seconds.')


def deep_dream_video(config):
    video_path = utils.parse_input_file(config['input'])
    tmp_input_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_input')
    tmp_output_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_out')
    config['dump_dir'] = tmp_output_dir
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)

    metadata = video_utils.extract_frames(video_path, tmp_input_dir)
    config['fps'] = metadata['fps']
    utils.print_deep_dream_video_header(config)

    last_img = None
    for frame_id, frame_name in enumerate(sorted(os.listdir(tmp_input_dir))):
        # Step 1: load the video frame
        print(f'Processing frame {frame_id}')
        frame_path = os.path.join(tmp_input_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config['img_width'])

        # Step 2: potentially blend it with the last frame
        if config['blend'] is not None and last_img is not None:
            # blend: 1.0 - use the current frame, 0.0 - use the last frame, everything in between will blend the two
            frame = utils.linear_blend(last_img, frame, config['blend'])

        # Step 3: Send the blended frame to some good old DeepDreaming
        dreamed_frame = deep_dream_static_image(config, frame)

        # Step 4: save the frame and keep the reference
        last_img = dreamed_frame
        dump_path = utils.save_and_maybe_display_image(config, dreamed_frame, name_modifier=frame_id)
        print(f'Saved DeepDream frame to: {os.path.relpath(dump_path)}\n')

    video_utils.create_video_from_intermediate_results(config)

    shutil.rmtree(tmp_input_dir)  # remove tmp files
    print(f'Deleted tmp frame dump directory {tmp_input_dir}.')


if __name__ == "__main__":

    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='figures.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use for dreaming", default=SupportedModels.VGG16_EXPERIMENTAL.name)
    parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
                        help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.IMAGENET.name)

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)

    # deep_dream_video_ouroboros specific arguments (ignore for other 2 functions)
    parser.add_argument("--create_ouroboros", action='store_true', help="Create Ouroboros video (default False)")
    parser.add_argument("--ouroboros_length", type=int, help="Number of video frames in ouroboros video", default=30)
    parser.add_argument("--fps", type=int, help="Number of frames per second", default=30)
    parser.add_argument("--frame_transform", choices=[t.name for t in TRANSFORMS],
                        help="Transform used to transform the output frame and feed it back to the network input",
                        default=TRANSFORMS.ZOOM_ROTATE.name)

    # deep_dream_video specific arguments (ignore for other 2 functions)
    parser.add_argument("--blend", type=float, help="Blend coefficient for video creation", default=0.85)

    # You usually won't need to change these as often
    parser.add_argument("--should_display", action='store_true', help="Display intermediate dreaming results (default False)")
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['dump_dir'] = OUT_VIDEOS_PATH if config['create_ouroboros'] else OUT_IMAGES_PATH
    config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
    config['input_name'] = os.path.basename(config['input'])

    # Create Ouroboros video (feeding neural network's output to it's input)
    if config['create_ouroboros']:
        deep_dream_video_ouroboros(config)

    # Create a blended DeepDream video
    elif any([config['input_name'].lower().endswith(video_ext) for video_ext in SUPPORTED_VIDEO_FORMATS]):  # only support mp4 atm
        deep_dream_video(config)

    else:  # Create a static DeepDream image
        print('Dreaming started!')
        img = deep_dream_static_image(config, img=None)  # img=None -> will be loaded inside of deep_dream_static_image
        dump_path = utils.save_and_maybe_display_image(config, img)
        print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')

