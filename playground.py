"""
    This file serves as a playground for understanding some of the concepts used
    in the development of the DeepDream algorithm.

"""

import time
import os
import enum


import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from torchvision import transforms


from utils.constants import *
import utils.utils as utils
import utils.video_utils as video_utils
from models.definitions.vggs import Vgg16


# Note: don't use scipy.ndimage it's way slower than OpenCV
def understand_frame_transform():
    """
        Pick different transform matrices here and see what they do.
    """
    height, width, num_channels = [500, 500, 3]
    s = 0.05

    # Create a white square on the black background
    img = np.zeros((height, width, num_channels))
    img[100:400, 100:400] = 1.0
    img_center = (width / 2, height / 2)

    # Translation
    tx, ty = [10, 5]
    translation_matrix = np.asarray([[1., 0., tx],
                                     [0., 1., ty],
                                     [0., 0., 1.]])

    # Rotation
    deg = 10  # rotation in degrees
    theta = (deg / 180) * np.pi  # convert to radians
    origin_rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0., 0., 1.]])

    # Does a similar thing to above but returns 2x3 matrix so just append the last row
    rotation_matrix = cv.getRotationMatrix2D(img_center, deg, scale=1.09)
    full_rotation_matrix = np.vstack([rotation_matrix, np.asarray([0., 0., 1.])])

    # Affine
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv.getAffineTransform(pts1, pts2)
    full_affine_matrix = np.vstack([affine_matrix, np.asarray([0., 0., 1.])])

    # Perspective
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    perspective_matrix = cv.getPerspectiveTransform(pts1, pts2)  # This one gives 3x3 transform matrix directly

    # This one was originally used and it represents diagonal values of the 3x3 projective matrix
    # You can't use it with OpenCV in this form
    zoom_matrix_simple = np.asarray([1 - s, 1 - s, 1])

    ts = time.time()  # start perf timer

    transformed_img = img
    for i in range(10):
        # transformed_img = nd.affine_transform(transformed_img, zoom_matrix, [height * s / 2, width * s / 2, 0], order=1)
        transformed_img = cv.warpPerspective(transformed_img, full_rotation_matrix, (width, height))
        plt.imshow(np.hstack([img, transformed_img])); plt.show()

    # And this kids is why you should use OpenCV
    # nd.affine_transform: ~720 ms
    # cv.warpPerspective: ~ 17 ms
    print(f'{(time.time()-ts)*1000} ms')  # result readout


def understand_blend():
    img1 = utils.load_image(os.path.join(INPUT_DATA_PATH, 'figures.jpg'), (500, 500))
    img2 = utils.load_image(os.path.join(INPUT_DATA_PATH, 'cloud.jpg'), (500, 500))

    for alpha in np.arange(0, 1.2, 0.2):
        blend = img1 + alpha * (img2 - img1)  # This is how PIL's blend works simple linear interpolation
        plt.imshow(blend)
        plt.show()


def understand_pytorch_gradients():
    """
        This builds up a computational graph in PyTorch the same way as a neural network does and is enough to understand
        why dst.diff[:] = dst.data (used in the original repo) is equivalent to MSE loss with sum reduction divided by 2.
        Most of the implementations use some form of MSE loss or L2, so it's worth understanding the equivalence.

        I found this blog super useful for understanding how automatic differentiation engine works in PyTorch:
        https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
    """
    def print_tensor_summary(t):  # helper function
        print(f'data={t.data}')
        print(f'requires_grad={t.requires_grad}')
        print(f'grad={t.grad}')
        print(f'grad_fn={t.grad_fn}')
        print(f'is_leaf={t.is_leaf}')

    x = torch.tensor([[-2.0, 1.0], [1.0, 3.0]], requires_grad=True)  # think of x as the input image
    y = x + 2  # some random processing builds up the computational graph in PyTorch
    z = y * y * 3

    z.backward(z)  # this one is equivalent to the commented out expression below

    # z is a matrix like this z = [[z11, z12], [z21, z22]] so doing MSE loss with sum reduction will give us this:
    # out = (z11^2 + z12^2 + z21^2 + z22^2) / 2 -> so dL/dz11 = z11 similarly for z12, z21, z22
    # that means that grad of z11 node will be populated with exactly z11 value (that's dL/dz11)
    # because the grad field of z11 should store dL/dz11 and that's the reason why z.backward(z) also works.

    # backward() implicitly passes torch.tensor(1.) as the argument,
    # because dL/L = 1 (derivative of loss with respect to loss equals 1)
    # out = torch.nn.MSELoss(reduction='sum')(z, torch.zeros_like(z)) / 2
    # out.backward()

    print_tensor_summary(x)  # Try both out and you will see that grad field of x ("the image") is the same

    # On calling backward(), gradients are populated only for the nodes which have both requires_grad and is_leaf True.
    # Here only x is both the leaf and has requires_grad set to true. Print other tensors and you'll see that grad=None.

    # The backward graph is created dynamically during the forward pass.
    # Graph consists of Function objects (the .grad_fn attribute of each torch.Tensor is an entry point into this graph)
    # Function class has 2 important member functions: 1) forward 2) backward which are called during forward/backprop

    # Take your time to understand this, it's actually really easy once it sinks in.


def deep_dream_simple(img_path, dump_path):
    """
        Contains the gist of DeepDream algorithm - takes 5 minutes to write down - if you know what you're doing.
        No support for: spatial shifting (aka jitter), octaves/image pyramid, clipping, gradient smoothing, etc.

        Most of the "code" are comments otherwise it literally takes 15 minutes to write down.

    """
    img = utils.load_image(img_path, target_shape=500)  # load numpy, [0, 1] image
    # Normalize image - VGG 16 and in general Pytorch (torchvision) models were trained like this,
    # so they learned to work with this particular distribution
    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1
    # Transform into PyTorch tensor, send to GPU and add dummy batch dimension. Models are expecting it, GPUs are
    # highly parallel computing machines so in general we'd like to process multiple images all at once
    # shape = (1, 3, H, W)
    img_tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    img_tensor.requires_grad = True  # set this to true so that PyTorch will start calculating gradients for img_tensor

    model = Vgg16(requires_grad=False).to(DEVICE)  # Instantiate VGG 16 and send it to GPU

    n_iterations = 10
    learning_rate = 0.3

    for iter in range(n_iterations):
        out = model(img_tensor)
        activations = out.relu4_3  # pick out particular feature maps (aka activations) that you're interested in
        activations.backward(activations)  # whatever is the biggest activation value make it even bigger

        img_tensor_grad = img_tensor.grad.data
        smooth_grads = img_tensor_grad / torch.std(img_tensor_grad)
        img_tensor.data += learning_rate * smooth_grads  # gradient ascent

        img_tensor.grad.data.zero_()  # clear the gradients otherwise they would get accumulated

    # Send the PyTorch tensor back to CPU, detach it from the computational graph, convert to numpy
    # and make it channel last format again (calling ToTensor converted it to channel-first format)
    img = np.moveaxis(img_tensor.to('cpu').detach().numpy()[0], 0, 2)
    img = (img * IMAGENET_STD_1) + IMAGENET_MEAN_1  # de-normalize
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    cv.imwrite(dump_path, img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...
    print(f'Saved naive deep dream image to {os.path.relpath(dump_path)}')


class PLAYGROUND(enum.Enum):
    GEOMETRIC_TRANSFORMS = 0,
    BLEND = 1,
    PT_GRADIENTS = 2,
    DEEPDREAM_NAIVE = 3,
    CREATE_GIF = 4


if __name__ == "__main__":
    # Pick the function you want to play with here
    playground_fn = PLAYGROUND.DEEPDREAM_NAIVE

    if playground_fn == PLAYGROUND.GEOMETRIC_TRANSFORMS:
        understand_frame_transform()

    elif playground_fn == PLAYGROUND.BLEND:
        understand_blend()

    elif playground_fn == PLAYGROUND.PT_GRADIENTS:
        understand_pytorch_gradients()

    elif playground_fn == PLAYGROUND.DEEPDREAM_NAIVE:
        img_path = os.path.join(INPUT_DATA_PATH, 'figures.jpg')
        dump_path = os.path.join(OUT_IMAGES_PATH, 'simple.jpg')
        deep_dream_simple(img_path, dump_path)

    elif playground_fn == PLAYGROUND.CREATE_GIF:
        # change this to a directory where you've saved the frames of interest
        input_frames_dir = os.path.join(DATA_DIR_PATH, 'input')
        video_utils.create_gif(input_frames_dir, os.path.join(OUT_GIF_PATH, 'default.gif'))

    else:
        raise Exception(f'{playground_fn} not supported!')

