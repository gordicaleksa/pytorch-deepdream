import time
import os


import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import torch
import cv2 as cv


import utils.utils as utils


# rotation:
# zoom: [1-s,1-s,1] [h*s/2,w*s/2,0]
# vertical stretch:  [1-s,1,1], [h*s/2,0,0]
# note: don't use scipy.ndimage it's way slower than OpenCV
# todo: make a set of interesting transforms in OpenCV (e.g. spiral-zoom motion)
def understand_affine():
    h, w, c = [500, 500, 3]
    s = 0.05

    img = np.zeros((h, w, c))
    img[100:400, 100:400] = 1.0

    matrix = np.asarray([0.95, 0.95, 1])

    transformed_img = img
    deg = 3
    theta = (deg / 180) * np.pi
    matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0., 0., 1.]])
    zoom_matrix = np.asarray([[1-s, 0, 0],
                        [0, 1-s, 0],
                        [0., 0., 1.]])
    ts = time.time()
    for i in range(10):
        transformed_img = nd.affine_transform(transformed_img, zoom_matrix, [h*s/2,w*s/2,0], order=1)
        # transformed_img = cv.warpPerspective(transformed_img, zoom_matrix, (w, h))
        # plt.imshow(np.hstack([img, transformed_img])); plt.show()

    print(f'{(time.time()-ts)*1000} ms')
    plt.imshow(np.hstack([img, transformed_img]));
    plt.show()


def tensor_summary(t):
    print(f'data={t.data}')
    print(f'requires_grad={t.requires_grad}')
    print(f'grad={t.grad}')
    print(f'grad_fn={t.grad_fn}')
    print(f'is_leaf={t.is_leaf}')


# todo: add playground function for understanding PyTorch gradients
# todo: explain that diff[:] is equivalent to taking MSE loss
def play_with_pytorch_gradients():


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


def understand_blend():
    input_images_path = os.path.join(os.path.dirname(__file__), 'data', 'input-images')
    img1 = utils.load_image(os.path.join(input_images_path, 'figures.jpg'), (500, 500))
    img2 = utils.load_image(os.path.join(input_images_path, 'cloud.jfif'), (500, 500))

    for alpha in np.arange(0, 1.2, 0.2):
        blend = img1 + alpha * (img2 - img1)
        plt.imshow(blend)
        plt.show()


def visualize_optical_flow():
    input_images_path = os.path.join(os.path.dirname(__file__), 'data', 'input-images')
    img1 = cv.cvtColor(utils.load_image(os.path.join(input_images_path, 'out_032.png')), cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(utils.load_image(os.path.join(input_images_path, 'out_033.png')), cv.COLOR_BGR2GRAY)

    # plt.imshow(np.hstack([img1, img2])); plt.show()

    flow = cv.calcOpticalFlowFarneback(img1, img2, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0, flow=None)

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((540, 960, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    plt.imshow(rgb);
    plt.show()


if __name__ == "__main__":





