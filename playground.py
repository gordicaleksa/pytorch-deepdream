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