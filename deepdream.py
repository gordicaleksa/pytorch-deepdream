import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd

# todo: add support for static input image
# todo: experiment with different models
# todo: experiment with different single/multiple layers
# todo: experiment with different objective functions (L2, guide, etc.)

from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
octave_scale = 1.4
c2 = nd.zoom(china, (1.0/octave_scale, 1.0/octave_scale, 1), order=1)
print(china.shape, c2.shape)

# print(np.percentile(china, 99.98))

a = np.arange(0, 9).reshape(3,3)
b = a + 1
print(a)

# plt.imshow(china)
# plt.show()
# plt.imshow(c2)
# plt.show()

jitter = 32

for i in range(3):
    print(i)

ox, oy = np.random.randint(-jitter, jitter+1, 2)

# china = np.roll(np.roll(china, ox, 1), oy, 2)
# plt.imshow(china)
# plt.show()

# todo: add support for video (simple affine transform)

