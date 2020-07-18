## Deep Dream :computer: + :ocean::zzz: = :heart:
This repo contains PyTorch implementation of the Deep Dream algorithm (:link: blog by [Mordvintstev et al.](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)).

And it will give you the power to create these weird, psychedelic-looking images:

<p align="center">
<img src="data/examples/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_12_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized500.jpg" width="488"/>
<img src="data/examples/lion_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu3_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_resized400.jpg" width="325"/>
</p>

Not bad, huh?

### What is Deep Dream algorithm?
In a nutshell the algorithm maximizes activations of chosen network layers (1 or multiple) by doing **gradient ascent**.

So from an input image like the one on the left after "dreaming" we get the image on the right:
<p align="center">
<img src="data/input/figures.jpg" width="400"/>
<img src="data/examples/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_4_pyrratio_1.4_iter_10_lr_0.09_shift_41.jpg" width="400"/>
</p>

Who would have said that neural networks had this creativity hidden inside? :art:

### Why yet another Deep Dream repo?

Most of the original Deep Dream repos were written in **Caffe** and the ones written in PyTorch are usually really hard to read and understand.

This repo is an attempt of making the **cleanest** DeepDream repo that I'm aware of + it's written in **PyTorch!** :heart:

## Examples

Here are some examples that you can create using this code.

### Lower layers = lower level features

By using shallower layers of neural networks you'll get lower level patterns (edges, circles, colors, etc.) as the output:

<p align="center">
<img src="data/examples/low_level_dreaming/figures_width_600_model_RESNET50_IMAGENET_layer2_pyrsize_11_pyrratio_1.3_iter_30_lr_0.09_shift_150_resized300.jpg" width="270"/>
<img src="data/examples/low_level_dreaming/figures_width_600_model_RESNET50_IMAGENET_layer2_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_34_resized300.jpg" width="270"/>
<img src="data/examples/low_level_dreaming/figures_width_600_model_GOOGLENET_IMAGENET_inception3b_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_35_resized300.jpg" width="270"/>
</p>

Here the first 2 images came from ResNet50 and the last one came from GoogLeNet (both pretrained on ImageNet).

### Higher Layers = High level features

By using deeper network layers you'll get higher level patterns (eyes, snouts, animal heads):

<p align="center">
<img src="data/examples/high_level_dreaming/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_12_pyrratio_1.4_iter_10_lr_0.09_shift_101_resized300.jpg" width="270"/>
<img src="data/examples/high_level_dreaming/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_34_resized300.jpg" width="270"/>
<img src="data/examples/high_level_dreaming/green_bridge_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_resized.jpg" width="270"/>
</p>

The 1st and 3rd were created using VGG 16 (ImageNet) and the middle one using ResNet50 pretrained on Places 365.

### Dataset matters (ImageNet vs Places 365)

If we keep every other parameter the same but we swap the pretrained weights we get these:

<p align="center">
<img src="data/examples/dataset_matters/figures_width_600_model_RESNET50_IMAGENET_layer4_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
<img src="data/examples/dataset_matters/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
</p>

Left: **ResNet50-ImageNet** (we can see more animal features) Right: **ResNet50-Places365** (human built stuff, etc.).

### Impact of increasing the pyramid size

Dreaming is performed on multiple image resolutions stacked "vertically" (we call that an **image pyramid**).

<p align="center">
<img src="data/examples/pyramid_size/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_3_pyrsize_3_pyrratio_1.3_iter_10_lr_0.09_shift_33_resized300.jpg" width="270"/>
<img src="data/examples/pyramid_size/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_3_pyrsize_7_pyrratio_1.3_iter_10_lr_0.09_shift_33_resized300.jpg" width="270"/>
<img src="data/examples/pyramid_size/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_3_pyrsize_9_pyrratio_1.3_iter_10_lr_0.09_shift_33_resized300.jpg" width="270"/>
</p>

Going from left to right the only parameter that changed was the pyramid size from left to right: 3, 7, 9 levels.

### Impact of increasing the pyramid ratio



*Note: all of the deepdream images were produced by me (using this repo), credits for original image artists [are given bellow](#acknowledgements).*

## Setup

1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` (while in project directory)
3. Run `activate pytorch-deepdream`

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

-----

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system. 

Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md) and use the most up-to-date versions of Miniconda and CUDA/cuDNN.
(I recommend CUDA 10.1 as it is compatible with PyTorch 1.5, which is used in this repo, and newest compatible cuDNN)

## Usage

*Note: All of the examples have parameters used to create them encoded into the file name, so you can either reconstruct them
or create new ones - although there is some randomness in the process so identical reconstructions are not guaranteed.*

### Debugging/Experimenting


## Acknowledgements

I found these repos useful (while developing this one):
* [deepdream](https://github.com/google/deepdream) (Caffe, original repo)
* [DeepDreamAnim](https://github.com/samim23/DeepDreamAnim) (Caffe)
* [AI-Art](https://github.com/Adi-iitd/AI-Art/blob/master/DeepDream.py) (PyTorch)
* [neural-dream](https://github.com/ProGamerGov/neural-dream) (PyTorch)
* [DeepDream](https://github.com/CharlesPikachu/DeepDream) (PyTorch)

I found the images I was using here:
* [awesome figures pic](https://www.pexels.com/photo/action-android-device-electronics-595804/)
* [awesome bridge pic](https://www.pexels.com/photo/gray-bridge-and-trees-814499/)

Other images are now already classics in the NST and DeepDream worlds.

## Citation

If you find this code useful for your research, please cite the following:

```
@misc{Gordić2020DeepDream,
  author = {Gordić, Aleksa},
  title = {pytorch-deepdream},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-deepdream}},
}
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/LICENCE)