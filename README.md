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

## Static Image Examples

Here are some examples that you can create using this code (**deep_dream_static_image** function).

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

Going from left to right the only parameter that changed was the pyramid size (from left to right: 3, 7, 9 levels).

### Impact of increasing the pyramid ratio

Playing with pyramid ratio has a similar/related effect - the basic idea is that the relative area of the image which the deeper neurons can modify and "see"
(the so-called **receptive field** of the net) is increasing and we get increasingly bigger features like eyes popping out (from left to right: 1.1, 1.5, 1.8):

<p align="center">
<img src="data/examples/pyramid_ratio/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_2_pyrsize_5_pyrratio_1.1_iter_10_lr_0.09_shift_38_resized300.jpg" width="270"/>
<img src="data/examples/pyramid_ratio/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_2_pyrsize_5_pyrratio_1.5_iter_10_lr_0.09_shift_38_resized300.jpg" width="270"/>
<img src="data/examples/pyramid_ratio/figures_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_2_pyrsize_5_pyrratio_1.8_iter_10_lr_0.09_shift_38_resized300.jpg" width="270"/>
</p>

*Note:* you can see the exact params used in the image filename itself located in `data/examples/pyramid_ratio/`.

## Ouroboros Examples

Here are some further examples that you can create using this code (**deep_dream_video_ouroboros** function).

The idea here is that whatever the network dreams just feed that back to it's input and apply geometric transform.

### Ouroboros: Zoom transform

If we apply only central zoom we get this:

<img src="data/examples/ouroboros/zoom.gif" />

### Ouroboros: Zoom and Rotation transforms

Applying central zoom and at the same time applying a 3 degree rotation per frame yields this:

<img src="data/examples/ouroboros/zoom_rotate.gif" />

### Ouroboros: Translation 

Finally if we do a simple translation (5 px per frame top left to bottom right direction):

<img src="data/examples/ouroboros/translation.gif" />

Hopefully these did not break your brain - it feels like web 1.0 early 2000s. Bear with me.

## DeepDream video examples

Instead of feeding the output back to input we just apply the algorithm per frame and apply some linear blending:

<img src="data/examples/deepdream_video/deepdream_video.gif" />

Linear blending just combines the current frame with the last one so as to reduce the flicker (here I used 0.85)

*Note: all of the deepdream images/GIFs were produced by me, credits for original image artists [are given bellow](#acknowledgements).*

## Setup

1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` (while in project directory)
3. Run `activate pytorch-deepdream`

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

*Note:* If you wish to use video functions I have - you'll need **ffmpeg** in your system path.

-----

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system. 

Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md) and use the most up-to-date versions of Miniconda and CUDA/cuDNN.

## Basic Usage

Important thing to note - whatever image or video you want to use **place it inside the `data/input/` directory.**

To create some **static Deep Dream images** run the following command:

`python deepdream.py --input <img_name>`

This will use the default settings but you'll immediately get a meaningful result saved to:<br/>
`data/out-images/VGG16_EXPERIMENTAL_IMAGENET/`<br/>
The last directory will change depending on the model and pretrained weights you use.

-----
To run **Ouroboros** do the following:

`python deepdream.py --input <img_name> --is_video true`

It will dump the intermediate frames to `data/out-videos/VGG16_EXPERIMENTAL_IMAGENET/` and it will save the final video to `data/out-videos`.

-----
To run **Deep Dream video** run this:

`python deepdream.py --input <video_name>`

It will dump the intermediate frames to `data/out-videos/tmp_out` and it will save the final video to `data/out-videos`.

## Experimenting

You'll probably wish to have more control of the output you create - and the code is hopefully self-explanatory to help you do that.
I'll just summarize the most important params here:

`--model` - choose between VGG 16 (best for high-level features), ResNet 50 (nice for mid-level features), GoogLeNet (low-to-mid features are nice).
AlexNet didn't give me nice results so I haven't used any of it's outputs in this README - if you manage to get it working please create an issue.

`--layers_to_use` - you can use single or multiple layers here just put them in a list like ['relu3_3', 'relu4_3']. <br/>
Depending on the model you choose you'll have to set different layer names:<br/>

For VGG16_EXPERIMENTAL you have these on your disposal: `relu3_3`, `relu4_1`, `relu4_2`, etc.<br/>
(checkout `models/definitions/vggs.py` for more details)

For RESNET50 `layer1`, `layer2`, `layer3` and `layer4` but again go to `models/definitions/resnets.py` and expose the layers
that you find particularly beautiful. There are many layers you can experiment with especially with ResNet50.

`--pyramid_size` - already briefly touched on this one - the bigger you go here the the less recognizable the original image will become.

`--pyramid_ratio` - some combinations of this one and `pyramid_size` will make too small of an output and crash the program.

You're ready to go! Here is some more candy for you:

<p align="center">
<img src="data/examples/footer/figures_width_600_model_RESNET50_PLACES_365_layer3_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
<img src="data/examples/footer/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
</p>

-----

*Note: All of the examples I used in this README have parameters used to create them encoded directly into the file name,<br/>
so you can reconstruct them - although there is some randomness in the process so identical reconstructions are not guaranteed.*

There is a small ambiguity on which exact sublayer was used (e.g. ResNet50's layer4 but which exact sublayer?)<br/>
I usually encoded the sublayer through the `shift_` infix you can just try out a couple of them and find the exact one.

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

Places 365 pretrained models came from [this awesome repo](https://github.com/CSAILVision/places365).

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