import time
import os


import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import torch
import cv2 as cv


import utils.utils as utils
import utils.video_utils as video_utils
from deepdream import gradient_ascent
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
    inputs_path = os.path.join(os.path.dirname(__file__), 'data', 'input')
    img1 = utils.load_image(os.path.join(inputs_path, 'figures.jpg'), (500, 500))
    img2 = utils.load_image(os.path.join(inputs_path, 'cloud.jpeg'), (500, 500))

    for alpha in np.arange(0, 1.2, 0.2):
        blend = img1 + alpha * (img2 - img1)  # This is how PIL's blend works simple linear interpolation
        plt.imshow(blend)
        plt.show()


# todo: add playground function for understanding PyTorch gradients
# todo: explain that diff[:] is equivalent to taking MSE loss
def understand_pytorch_gradients():
    def tensor_summary(t):
        print(f'data={t.data}')
        print(f'requires_grad={t.requires_grad}')
        print(f'grad={t.grad}')
        print(f'grad_fn={t.grad_fn}')
        print(f'is_leaf={t.is_leaf}')

    x = torch.tensor([[-2.0, 1.0], [1.0, 1.0]], requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    out.backward()

    tensor_summary(x)
    tensor_summary(y)
    tensor_summary(z)
    tensor_summary(out)

    # On calling backward(), gradients are populated only for the nodes which have both requires_grad and is_leaf True.
    # Remember, the backward graph is already made dynamically during the forward pass.
    # graph of Function objects (the .grad_fn attribute of each torch.Tensor is an entry point into this graph)
    # Function class ha 2 member functions: 1) forward 2) backward
    # whatever comes from the front layers to current node is saved in grad attribute of the current node
    # backward is usually called on L-node with unit tensor because dL/L = 1


def deep_dream_simple(img_path):
    """
        Contains the gist of DeepDream algorithm - takes 15 minutes to write down.
        No support for: spatial jitter/shifting, octaves/image pyramid, clipping policy, gradient normalization.
    """
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


if __name__ == "__main__":
    understand_frame_transform()

    # frames_dir = r'C:\tmp_data_dir\YouTube\CodingProjects\DeepDream\data\out-videos\tmp'
    # out_path = r'C:\tmp_data_dir\YouTube\CodingProjects\DeepDream\data\out-videos\tmp\first.gif'
    # video_utils.create_gif(frames_dir, out_path)
