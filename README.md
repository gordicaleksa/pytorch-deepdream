## Deep Dream :computer: + :ocean::zzz: = :heart:
This repo contains a PyTorch implementation of the Deep Dream algorithm (:link: blog by [Mordvintstev et al.](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)).

It's got a full support for the **command line** usage and a **Jupyter Notebook**!

And it will give you the power to create these weird, psychedelic-looking images:

<p align="center">
<img src="data/examples/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_12_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized500.jpg" width="488"/>
<img src="data/examples/lion_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu3_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_resized400.jpg" width="325"/>
<img 
src="data/examples/figures_dimensions_600_model_RN50_CLIP_OPENAI_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg">
</p>

Not bad, huh?

I strongly suggest you start with the [Jupyter notebook](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/The%20Annotated%20DeepDream.ipynb) that I've created!

*Note: it's pretty large, ~10 MBs, so it may take a couple of attempts to load it in the browser here on GitHub.*

## Table of Contents
* [What is DeepDream?](#what-is-deepdream-algorithm)
    * [Image visualizations and experiments](#static-image-examples)
    * [Ouroboros video examples](#ouroboros-video-examples)
    * [DeepDream video examples](#deepdream-video-examples)
* [Setup](#setup)
* [Usage](#usage)
* [Hardware requirements](#hardware-requirements)
* [Learning material](#learning-material)

### What is DeepDream algorithm?
In a nutshell the algorithm maximizes the activations of chosen network layers by doing a **gradient ascent**.

So from an input image like the one on the left after "dreaming" we get the image on the right:
<p align="center">
<img src="data/input/figures.jpg" width="400"/>
<img src="data/examples/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_4_pyrratio_1.4_iter_10_lr_0.09_shift_41.jpg" width="400"/>
</p>

Who would have said that neural networks had this creativity hidden inside? :art:

#### Why yet another Deep Dream repo?

Most of the original Deep Dream repos were written in **Caffe** and the ones written in PyTorch are usually really hard to read and understand.
This repo is an attempt of making the **cleanest** DeepDream repo that I'm aware of + it's written in **PyTorch!** :heart:
Update: Also most of the other repos are focused on deep dreaming Convolutional Neural Networks (CNNs). There was not much resource experimenting with Vision Transformers (ViT) model variants and the vision module of CLIP variants. The visual patterns that would emerge in pretrained CNNs on image recognition tasks has been studied extensively but we were curious to observe and compare the visual artifacts that would emerge by deep dreaming ViTs and CLIP variants. 

## Static Image Examples

Here are some examples that you can create using this code!

### Optimizing shallower layers = Amplify low-level features

By using shallower layers of CNN neural networks you'll get lower level patterns (edges, circles, colors, etc.) as the output:

<p align="center">
<img src="data/examples/low_level_dreaming/figures_width_600_model_RESNET50_IMAGENET_layer2_pyrsize_11_pyrratio_1.3_iter_30_lr_0.09_shift_150_resized300.jpg" width="270"/>
<img src="data/examples/low_level_dreaming/figures_width_600_model_RESNET50_IMAGENET_layer2_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_34_resized300.jpg" width="270"/>
<img src="data/examples/low_level_dreaming/figures_width_600_model_GOOGLENET_IMAGENET_inception3b_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_35_resized300.jpg" width="270"/>
</p>

Here the first 2 images came from ResNet50 and the last one came from GoogLeNet (both pretrained on ImageNet).

### Optimizing deeper Layers = Amplify high-level features

By using deeper network layers you'll get higher level patterns (eyes, snouts, animal heads):

<p align="center">
<img src="data/examples/high_level_dreaming/figures_width_600_model_VGG16_IMAGENET_relu4_3_pyrsize_12_pyrratio_1.4_iter_10_lr_0.09_shift_101_resized300.jpg" width="270"/>
<img src="data/examples/high_level_dreaming/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_34_resized300.jpg" width="270"/>
<img src="data/examples/high_level_dreaming/green_bridge_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_resized.jpg" width="270"/>
</p>

The 1st and 3rd were created using VGG 16 (ImageNet) and the middle one using ResNet50 pretrained on Places 365.


### Optimizing visual and text prompt similarity (CLIP models)
By passing a textual prompt through a CLIP model, the final similarity logits to a given image can be maximized. This can open a possibility of injecting infinite concepts and artifacts inside an image!

<p align="center">
<img src="data/examples/prompt_deepdreaming/whales.jpg" width="320"/>
<img src="data/examples/prompt_deepdreaming/squares and triangles.jpg" width="320"/>
   <img src="data/examples/prompt_deepdreaming/NewYork City.jpg" width="320"/>
</p>

**OpenCLIP ConvNext-base-320** model with the following textual prompts from left to right: **Whales**, **Squares and Triangles**, and **NewYork City**. 

### Dataset matters 

If we keep every other parameter the same but we swap the pretrained weights we get these:

<p align="center">
<img src="data/examples/dataset_matters/figures_width_600_model_RESNET50_IMAGENET_layer4_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
<img src="data/examples/dataset_matters/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
</p>

Left: **ResNet50-ImageNet** (we can see more animal features) Right: **ResNet50-Places365** (human built stuff, etc.).


For ResNet50, changing the pretraining datasets and training objectives (image recognition vs CLIP's pretraining) will yield:

<p align="center">
<img src="data/examples/dataset_matters/figures_dimensions_600_model_RN50_CLIP_CC12M_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="170"/>
<img src="data/examples/dataset_matters/figures_dimensions_600_model_RN50_CLIP_OPENAI_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="170"/>
<img  src="data/examples/dataset_matters/figures_dimensions_600_model_RN50_CLIP_YFCC15M_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="170"/>
<img   src="data/examples/dataset_matters/figures_dimensions_600_model_RN50_IMAGENET_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="170"/>
<img   src="data/examples/dataset_matters/figures_dimensions_600_model_RN50_PLACES_365_layer2_layer3_pyrsize_8_pyrratio_1.4_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="170"/>
</p>
From left to right: ResNet50-CC12M, ResNet50-OpenAI, ResNet50-YFCC15M, ResNet50-ImageNet, ResNet50-Places365.

### Impact of increasing the pyramid size

Dreaming is performed on multiple image resolutions stacked "vertically" (we call this an **image pyramid**).

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

**Note: you can see the exact params used to create these images encoded into the filename!**

Make sure to check out the [Jupyter notebook!](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/The%20Annotated%20DeepDream.ipynb), I've explained this thoroughly.

## Ouroboros Video Examples

Here are some further examples that you can create using this code!

The idea here is that whatever the network dreams just feed that back to it's input and apply a geometric transformation.

### Ouroboros: Zoom transform

If we apply only central zoom we get this:

<img src="data/examples/ouroboros/zoom.gif" />

### Ouroboros: Zoom and Rotation transforms

Applying central zoom and at the same time applying a 3 degree rotation per frame yields this:

<p align="left">
<img src="data/examples/ouroboros/zoom_rotate.gif" width="270"/>

<img src="data/examples/ouroboros/ouroboros_video_fps_30_figures_dimensions_400_model_RN50_CLIP_OPENAI_layer2_layer3_pyrsize_3_pyrratio_1.2_iter_1_lr_0.09_shift_32_smooth_0.5.gif" width="270"/>
</p>


### Ouroboros: Translation 

Finally if we do a simple translation (5 px per frame top left to bottom right direction):

<img src="data/examples/ouroboros/translation.gif" />

Hopefully these did not break your brain - it feels like web 1.0 early 2000s. Bear with me.

## DeepDream Video Examples

Instead of feeding the output back to input we just apply the algorithm per frame and apply some linear blending:

<img src="data/examples/deepdream_video/deepdream_video.gif" />

Linear blending just combines the current frame with the last one so as to reduce the flicker (here I used 0.85)

*Note: all of the deepdream images/GIFs were produced by me, credits for original image artists [are given bellow](#acknowledgements).*

## Setup

1. `git clone https://github.com/gordicaleksa/pytorch-deepdream`
1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `activate pytorch-deepdream` (for running scripts from your console or setup the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

*Note:* If you wish to use video functions I have - you'll need **ffmpeg** in your system path.

-----

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer as a way to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

## Usage

#### Option 1: Jupyter Notebook

Just run `jupyter notebook` from you Anaconda console and it will open up a session in your default browser. <br/>
Open `The Annotated DeepDream.ipynb` and you're ready to play!

**Note:** if you get `DLL load failed while importing win32api: The specified module could not be found` <br/>
Just do `pip uninstall pywin32` and then either `pip install pywin32` or `conda install pywin32` [should fix it](https://github.com/jupyter/notebook/issues/4980)!

#### Option 2: Use your IDE of choice

You just need to link the Python environment you created in the [setup](#setup) section.

#### Option 3: Command line

Navigate to/activate your env if you're using Anaconda (and I hope you do) and you can use the commands I've linked below.

---

**Tip: Place your images/videos inside the `data/input/` directory and you can then just reference 
your files (images/videos) by their name instead of using absolute/relative paths.**


## Available models and pretraining datasets


| Model | Pretraining Dataset |
| --- | --- |
| VGG16 | IMAGENET |
| VGG16_EXPERIMENTAL | IMAGENET |
| GOOGLENET | IMAGENET |
| RESNET50 | IMAGENET, PLACES_365 |
| ALEXNET | IMAGENET, PLACES_365 |
| VIT_B_16 | IMAGENET, CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_S34B_B88K, CLIP_DATACOMP_L_S1B_B8K, CLIP_COMMONPOOL_L_CLIP_S1B_B8K, CLIP_COMMONPOOL_L_LAION_S1B_B8K, CLIP_COMMONPOOL_L_IMAGE_S1B_B8K, CLIP_COMMONPOOL_L_TEXT_S1B_B8K, CLIP_COMMONPOOL_L_BASIC_S1B_B8K, CLIP_COMMONPOOL_L_S1B_B8K |
| VIT_B_32 | IMAGENET, CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_E16, CLIP_LAION2B_S34B_B79K, CLIP_DATACOMP_M_S128M_B4K, CLIP_COMMONPOOL_M_CLIP_S128M_B4K, CLIP_COMMONPOOL_M_LAION_S128M_B4K, CLIP_COMMONPOOL_M_IMAGE_S128M_B4K, CLIP_COMMONPOOL_M_TEXT_S128M_B4K, CLIP_COMMONPOOL_M_BASIC_S128M_B4K, CLIP_COMMONPOOL_M_S128M_B4K, CLIP_DATACOMP_S_S13M_B4K, CLIP_COMMONPOOL_S_CLIP_S13M_B4K, CLIP_COMMONPOOL_S_LAION_S13M_B4K, CLIP_COMMONPOOL_S_IMAGE_S13M_B4K, CLIP_COMMONPOOL_S_TEXT_S13M_B4K, CLIP_COMMONPOOL_S_BASIC_S13M_B4K, CLIP_COMMONPOOL_S_S13M_B4K |
| VIT_L_16 | IMAGENET |
| VIT_L_32 | IMAGENET |
| VIT_L_14 | CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_S32B_B82K, CLIP_DATACOMP_XL_S13B_B90K, CLIP_COMMONPOOL_XL_CLIP_S13B_B90K, CLIP_COMMONPOOL_XL_LAION_S13B_B90K, CLIP_COMMONPOOL_XL_S13B_B90K |
| VIT_L_14_336 | CLIP_OPENAI |
| RN50 | IMAGENET, PLACES_365, CLIP_OPENAI, CLIP_YFCC15M, CLIP_CC12M |
| RN101 | IMAGENET, CLIP_OPENAI, CLIP_YFCC15M |
| RN152 | IMAGENET |
| CONVNEXT_BASE | IMAGENET, CLIP_LAION400M_S13B_B51K, CLIP_LAION2B_S13B_B82K, CLIP_LAION2B_S13B_B82K_AUGREG, CLIP_LAION_AESTHETIC_S13B_B82K |
| CONVNEXT_XXLARGE | CLIP_LAION2B_S34B_B82K_AUGREG, CLIP_LAION2B_S34B_B82K_AUGREG_REWIND, CLIP_LAION2B_S34B_B82K_AUGREG_SOUP |
| CONVNEXT_LARGE | IMAGENET, CLIP_LAION2B_S26B_B102K_AUGREG, CLIP_LAION2B_S29B_B131K_FT, CLIP_LAION2B_S29B_B131K_FT_SOUP |
| CLIP_VIT_B_32 | CLIP_OPENAI |
| CLIP_VIT_B_16 | CLIP_OPENAI |
| CLIP_VIT_L_14 | CLIP_OPENAI |
| CLIP_VIT_L_14_336 | CLIP_OPENAI |
| CLIP_RN50 | CLIP_OPENAI |
| CLIP_RN101 | CLIP_OPENAI |
| CLIP_RN50x4 | CLIP_OPENAI |
| CLIP_RN50x16 | CLIP_OPENAI |
| CLIP_RN50x64 | CLIP_OPENAI |
| OPENCLIP_COCA_VIT_B_32 | CLIP_LAION2B_S13B_B90K, CLIP_MSCOCO_FINETUNED_LAION2B_S13B_B90K |
| OPENCLIP_COCA_VIT_L_14 | CLIP_LAION2B_S13B_B90K, CLIP_MSCOCO_FINETUNED_LAION2B_S13B_B90K |
| OPENCLIP_CONVNEXT_BASE | CLIP_LAION400M_S13B_B51K |
| OPENCLIP_CONVNEXT_BASE_W | CLIP_LAION2B_S13B_B82K, CLIP_LAION2B_S13B_B82K_AUGREG, CLIP_LAION_AESTHETIC_S13B_B82K |
| OPENCLIP_CONVNEXT_BASE_W_320 | CLIP_LAION_AESTHETIC_S13B_B82K, CLIP_LAION_AESTHETIC_S13B_B82K_AUGREG |
| OPENCLIP_CONVNEXT_LARGE_D | CLIP_LAION2B_S26B_B102K_AUGREG |
| OPENCLIP_CONVNEXT_LARGE_D_320 | CLIP_LAION2B_S29B_B131K_FT, CLIP_LAION2B_S29B_B131K_FT_SOUP |
| OPENCLIP_CONVNEXT_XXLARGE | CLIP_LAION2B_S34B_B82K_AUGREG, CLIP_LAION2B_S34B_B82K_AUGREG_REWIND, CLIP_LAION2B_S34B_B82K_AUGREG_SOUP |
| OPENCLIP_EVA01_G_14 | CLIP_LAION400M_S11B_B41K |
| OPENCLIP_EVA01_G_14_PLUS | CLIP_MERGED2B_S11B_B114K |
| OPENCLIP_EVA02_B_16 | CLIP_MERGED2B_S8B_B131K |
| OPENCLIP_EVA02_E_14 | CLIP_LAION2B_S4B_B115K |
| OPENCLIP_EVA02_E_14_PLUS | CLIP_LAION2B_S9B_B144K |
| OPENCLIP_EVA02_L_14 | CLIP_MERGED2B_S4B_B131K |
| OPENCLIP_EVA02_L_14_336 | CLIP_MERGED2B_S6B_B61K |
| OPENCLIP_RN50 | CLIP_OPENAI, CLIP_YFCC15M, CLIP_CC12M |
| OPENCLIP_RN50_QUICKGELU | CLIP_OPENAI, CLIP_YFCC15M, CLIP_CC12M |
| OPENCLIP_RN50X4 | CLIP_OPENAI |
| OPENCLIP_RN50X16 | CLIP_OPENAI |
| OPENCLIP_RN50X64 | CLIP_OPENAI |
| OPENCLIP_RN101 | CLIP_OPENAI, CLIP_YFCC15M |
| OPENCLIP_RN101_QUICKGELU | CLIP_OPENAI, CLIP_YFCC15M |
| OPENCLIP_ROBERTA_VIT_B_32 | CLIP_LAION2B_S12B_B32K |
| OPENCLIP_VIT_B_16 | CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_S34B_B88K, CLIP_DATACOMP_L_S1B_B8K, CLIP_COMMONPOOL_L_CLIP_S1B_B8K, CLIP_COMMONPOOL_L_LAION_S1B_B8K, CLIP_COMMONPOOL_L_IMAGE_S1B_B8K, CLIP_COMMONPOOL_L_TEXT_S1B_B8K, CLIP_COMMONPOOL_L_BASIC_S1B_B8K, CLIP_COMMONPOOL_L_S1B_B8K |
| OPENCLIP_VIT_B_16_PLUS_240 | CLIP_LAION400M_E31, CLIP_LAION400M_E32 |
| OPENCLIP_VIT_B_32 | CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_E16, CLIP_LAION2B_S34B_B79K, CLIP_DATACOMP_M_S128M_B4K, CLIP_COMMONPOOL_M_CLIP_S128M_B4K, CLIP_COMMONPOOL_M_LAION_S128M_B4K, CLIP_COMMONPOOL_M_IMAGE_S128M_B4K, CLIP_COMMONPOOL_M_TEXT_S128M_B4K, CLIP_COMMONPOOL_M_BASIC_S128M_B4K, CLIP_COMMONPOOL_M_S128M_B4K, CLIP_DATACOMP_S_S13M_B4K, CLIP_COMMONPOOL_S_CLIP_S13M_B4K, CLIP_COMMONPOOL_S_LAION_S13M_B4K, CLIP_COMMONPOOL_S_IMAGE_S13M_B4K, CLIP_COMMONPOOL_S_TEXT_S13M_B4K, CLIP_COMMONPOOL_S_BASIC_S13M_B4K, CLIP_COMMONPOOL_S_S13M_B4K |
| OPENCLIP_VIT_B_32_QUICKGELU | CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32 |
| OPENCLIP_VIT_BIGG_14 | CLIP_LAION2B_S39B_B160K |
| OPENCLIP_VIT_G_14 | CLIP_LAION2B_S12B_B42K, CLIP_LAION2B_S34B_B88K |
| OPENCLIP_VIT_H_14 | CLIP_LAION2B_S32B_B79K |
| OPENCLIP_VIT_L_14 | CLIP_OPENAI, CLIP_LAION400M_E31, CLIP_LAION400M_E32, CLIP_LAION2B_S32B_B82K, CLIP_DATACOMP_XL_S13B_B90K, CLIP_COMMONPOOL_XL_CLIP_S13B_B90K, CLIP_COMMONPOOL_XL_LAION_S13B_B90K, CLIP_COMMONPOOL_XL_S13B_B90K |
| OPENCLIP_VIT_L_14_336 | CLIP_OPENAI |
| OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32 | CLIP_LAION5B_S13B_B90K |
| OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14 | CLIP_FROZEN_LAION5B_S13B_B90K |


### DeepDream images

To create some **static Deep Dream images** run the following command:

`python deepdream.py --input <img_name> --img_dimensions 600`

This will use the default settings but you'll immediately get a meaningful result saved to:

`data/out-images/VGG16_EXPERIMENTAL_IMAGENET/`

Update: You can experiment deepdreaming different layers of different models specifying also their pretraining dataset and also text prompt deepdreaming for Openai and OpenCLIP CLIP models if available. As an example:

`! python deepdream.py --input <img_name> --layers_to_use "logits_per_image" --text_prompt "monkey peeling a banana" --model_name "CLIP_VIT_B_16" --pretrained_weights "CLIP_OPENAI" --num_gradient_ascent_iterations 100 --pyramid_ratio 1.2 --pyramid_size 3`

The result will be saved to: 

`data/out-images/CLIP_VIT_B_16_CLIP_OPENAI/golden_gate_dimensions_(224, 224)_model_CLIP_VIT_B_16_CLIP_OPENAI_logits_per_image_pyrsize_3_pyrratio_1.2_iter_100_lr_0.09_shift_32_smooth_0.5.jpg`

*Note: the output directory will change depending on the model and pretrained weights you use.*

### Ouroboros videos

To get the out-of-the-box **Ouroboros** 30-frame video do the following:

`python deepdream.py --input <img_name> --create_ouroboros --ouroboros_length 30`

It will dump the intermediate frames to `data/out-videos/VGG16_EXPERIMENTAL_IMAGENET/` and it will save the final video to `data/out-videos`.

### DeepDream videos

To create a **Deep Dream video** run this command:

`python deepdream.py --input <mp4 video name>`

It will dump the intermediate frames to `data/out-videos/tmp_out` and it will save the final video to `data/out-videos`.

---

Well, enjoy playing with this project! Here are some additional, beautiful, results:

<p align="center">
<img src="data/examples/footer/figures_width_600_model_RESNET50_PLACES_365_layer3_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
<img src="data/examples/footer/figures_width_600_model_RESNET50_PLACES_365_layer4_pyrsize_11_pyrratio_1.3_iter_10_lr_0.09_shift_32_resized400.jpg" width="400"/>
</p>

## Hardware requirements

A GPU with 2+ GBs will be more than enough.

You'll be able to create DeepDream images, Ouroboros and DeepDream videos.

If you don't have a GPU, the code will automatically run on the CPU but somewhat slower (especially for videos).

## Learning material

If you're having difficulties understanding DeepDream I did an overview of the algorithm [in this video](https://www.youtube.com/watch?v=6rVrh5gnpwk):

<p align="left">
<a href="https://www.youtube.com/watch?v=6rVrh5gnpwk" target="_blank"><img src="https://img.youtube.com/vi/6rVrh5gnpwk/0.jpg" 
alt="The GAT paper explained" width="480" height="360" border="10" /></a>
</p>

And also the [Jupyter Notebook](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/The%20Annotated%20DeepDream.ipynb) I created is the best place to start!

## Acknowledgements

I found these repos useful (while developing this one):
* [deepdream](https://github.com/google/deepdream) (Caffe, original repo)
* [DeepDreamAnim](https://github.com/samim23/DeepDreamAnim) (Caffe)
* [AI-Art](https://github.com/Adi-iitd/AI-Art/blob/master/DeepDream.py) (PyTorch)
* [neural-dream](https://github.com/ProGamerGov/neural-dream) (PyTorch)
* [DeepDream](https://github.com/CharlesPikachu/DeepDream) (PyTorch)
* [CLIP](https://github.com/openai/CLIP) (PyTorch)
* [OpenCLIP](https://github.com/mlfoundations/open_clip) (PyTorch)
    
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

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:
* Subscribing to my YouTube channel [The AI Epiphany](https://www.youtube.com/c/TheAiEpiphany) :bell:
* Follow me on [LinkedIn](https://www.linkedin.com/in/aleksagordic/) and [Twitter](https://twitter.com/gordic_aleksa) :bulb:
* Follow me on [Medium](https://gordicaleksa.medium.com/) :books: :heart:

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/LICENCE)
