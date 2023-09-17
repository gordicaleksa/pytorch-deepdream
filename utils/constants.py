import enum
import os


import numpy as np
import torch

import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

class ConstantsContext:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    ACTIVE_MEAN = IMAGENET_MEAN
    ACTIVE_STD = IMAGENET_STD
    LOWER_IMAGE_BOUND = torch.tensor((-ACTIVE_MEAN / ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
    UPPER_IMAGE_BOUND = torch.tensor(((1 - ACTIVE_MEAN) / ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
    
    @classmethod
    def use_imagenet(cls):
        cls.ACTIVE_MEAN = cls.IMAGENET_MEAN
        cls.ACTIVE_STD = cls.IMAGENET_STD
        cls._update_bounds()

    @classmethod
    def use_clip(cls):
        cls.ACTIVE_MEAN = cls.CLIP_MEAN
        cls.ACTIVE_STD = cls.CLIP_STD
        cls._update_bounds()
        
    @classmethod
    def _update_bounds(cls):
        cls.LOWER_IMAGE_BOUND = torch.tensor((-cls.ACTIVE_MEAN / cls.ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
        cls.UPPER_IMAGE_BOUND = torch.tensor(((1 - cls.ACTIVE_MEAN) / cls.ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
        

class TRANSFORMS(enum.Enum):
    ZOOM = 0
    ZOOM_ROTATE = 1
    TRANSLATE = 2


class SupportedModels(enum.Enum):
    # Vision Only Models - CNN
    VGG16  =  0
    VGG16_EXPERIMENTAL  =  1
    GOOGLENET  =  2
    RESNET50_EXPERIMENTAL  =  3
    RN50  =  4
    RN101  =  5
    RN152  =  6
    ALEXNET  =  7
    CONVNEXT_BASE  =  8
    CONVNEXT_LARGE  =  9
    CONVNEXT_XXLARGE  =  10
    # Vision Only Models - ViT
    VIT_B_16  =  11
    VIT_B_32  =  12
    VIT_L_14  =  13
    VIT_L_16  =  14
    VIT_L_32  =  15
    VIT_L_14_336  =  16
    # OpenAI CLIP Models
    CLIP_VIT_B_32  =  17
    CLIP_VIT_B_16  =  18
    CLIP_VIT_L_14  =  19
    CLIP_VIT_L_14_336  =  20
    CLIP_RN50  =  21
    CLIP_RN101  =  22
    CLIP_RN50x4  =  23
    CLIP_RN50x16  =  24
    CLIP_RN50x64  =  25
    # OpenCLIP Models
    OPENCLIP_COCA_VIT_B_32  =  26
    OPENCLIP_COCA_VIT_L_14  =  27
    OPENCLIP_CONVNEXT_BASE  =  28
    OPENCLIP_CONVNEXT_BASE_W  =  29
    OPENCLIP_CONVNEXT_BASE_W_320  =  30
    OPENCLIP_CONVNEXT_LARGE_D  =  31
    OPENCLIP_CONVNEXT_LARGE_D_320  =  32
    OPENCLIP_CONVNEXT_XXLARGE  =  33
    OPENCLIP_EVA01_G_14  =  34
    OPENCLIP_EVA01_G_14_PLUS  =  35
    OPENCLIP_EVA02_B_16  =  36
    OPENCLIP_EVA02_E_14  =  37
    OPENCLIP_EVA02_E_14_PLUS  =  38
    OPENCLIP_EVA02_L_14  =  39
    OPENCLIP_EVA02_L_14_336  =  40
    OPENCLIP_RN50  =  41
    OPENCLIP_RN50_QUICKGELU  =  42
    OPENCLIP_RN50X4  =  43
    OPENCLIP_RN50X16  =  44
    OPENCLIP_RN50X64  =  45
    OPENCLIP_RN101  =  46
    OPENCLIP_RN101_QUICKGELU  =  47
    OPENCLIP_ROBERTA_VIT_B_32  =  48
    OPENCLIP_VIT_B_16  =  49
    OPENCLIP_VIT_B_16_PLUS_240  =  50
    OPENCLIP_VIT_B_32  =  51
    OPENCLIP_VIT_B_32_QUICKGELU  =  52
    OPENCLIP_VIT_BIGG_14  =  53
    OPENCLIP_VIT_G_14  =  54
    OPENCLIP_VIT_H_14  =  55
    OPENCLIP_VIT_L_14  =  56
    OPENCLIP_VIT_L_14_336  =  57
    OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32  =  58
    OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14  =  59

class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1
    CLIP_CC12M = 2
    CLIP_COMMONPOOL_L_BASIC_S1B_B8K = 3
    CLIP_COMMONPOOL_L_CLIP_S1B_B8K = 4
    CLIP_COMMONPOOL_L_IMAGE_S1B_B8K = 5
    CLIP_COMMONPOOL_L_LAION_S1B_B8K = 6
    CLIP_COMMONPOOL_L_S1B_B8K = 7
    CLIP_COMMONPOOL_L_TEXT_S1B_B8K = 8
    CLIP_COMMONPOOL_M_BASIC_S128M_B4K = 9
    CLIP_COMMONPOOL_M_CLIP_S128M_B4K = 10
    CLIP_COMMONPOOL_M_IMAGE_S128M_B4K = 11
    CLIP_COMMONPOOL_M_LAION_S128M_B4K = 12
    CLIP_COMMONPOOL_M_S128M_B4K = 13
    CLIP_COMMONPOOL_M_TEXT_S128M_B4K = 14
    CLIP_COMMONPOOL_S_BASIC_S13M_B4K = 15
    CLIP_COMMONPOOL_S_CLIP_S13M_B4K = 16
    CLIP_COMMONPOOL_S_IMAGE_S13M_B4K = 17
    CLIP_COMMONPOOL_S_LAION_S13M_B4K = 18
    CLIP_COMMONPOOL_S_S13M_B4K = 19
    CLIP_COMMONPOOL_S_TEXT_S13M_B4K = 20
    CLIP_COMMONPOOL_XL_CLIP_S13B_B90K = 21
    CLIP_COMMONPOOL_XL_LAION_S13B_B90K = 22
    CLIP_COMMONPOOL_XL_S13B_B90K = 23
    CLIP_DATACOMP_L_S1B_B8K = 24
    CLIP_DATACOMP_M_S128M_B4K = 25
    CLIP_DATACOMP_S_S13M_B4K = 26
    CLIP_DATACOMP_XL_S13B_B90K = 27
    CLIP_FROZEN_LAION5B_S13B_B90K = 28
    CLIP_LAION2B_E16 = 29
    CLIP_LAION2B_S12B_B32K = 30
    CLIP_LAION2B_S12B_B42K = 31
    CLIP_LAION2B_S13B_B82K = 32
    CLIP_LAION2B_S13B_B82K_AUGREG = 33
    CLIP_LAION2B_S13B_B90K = 34
    CLIP_LAION2B_S26B_B102K_AUGREG = 35
    CLIP_LAION2B_S29B_B131K_FT = 36
    CLIP_LAION2B_S29B_B131K_FT_SOUP = 37
    CLIP_LAION2B_S32B_B79K = 38
    CLIP_LAION2B_S32B_B82K = 39
    CLIP_LAION2B_S34B_B79K = 40
    CLIP_LAION2B_S34B_B82K_AUGREG = 41
    CLIP_LAION2B_S34B_B82K_AUGREG_REWIND = 42
    CLIP_LAION2B_S34B_B82K_AUGREG_SOUP = 43
    CLIP_LAION2B_S34B_B88K = 44
    CLIP_LAION2B_S39B_B160K = 45
    CLIP_LAION2B_S4B_B115K = 46
    CLIP_LAION2B_S9B_B144K = 47
    CLIP_LAION400M_E31 = 48
    CLIP_LAION400M_E32 = 49
    CLIP_LAION400M_S11B_B41K = 50
    CLIP_LAION400M_S13B_B51K = 51
    CLIP_LAION5B_S13B_B90K = 52
    CLIP_LAION_AESTHETIC_S13B_B82K = 53
    CLIP_LAION_AESTHETIC_S13B_B82K_AUGREG = 54
    CLIP_MERGED2B_S11B_B114K = 55
    CLIP_MERGED2B_S4B_B131K = 56
    CLIP_MERGED2B_S6B_B61K = 57
    CLIP_MERGED2B_S8B_B131K = 58
    CLIP_MSCOCO_FINETUNED_LAION2B_S13B_B90K = 59
    CLIP_OPENAI = 60
    CLIP_YFCC15M = 61


FixedImageResolutionClasses = ["ViT_base", "ViT_large", "CLIP", "OpenCLIP"]


## Some models have a fixed input size due to the Attention layers.
## The CNN listed below have an attention pooling layer which require the model to have a fixed input size
FixedImageResolutions = {
    ## Vision Only Models - ViT
    "VIT_B_16" : (224,224), 
    "VIT_B_32" : (224,224), 
    "VIT_L_14" : (224,224), 
    "VIT_L_16" : (224,224), 
    "VIT_L_32" : (224,224), 
    "VIT_L_14_336" : (336,336), 
    ## OpenAI CLIP Models
    "CLIP_VIT_B_32" : (224,224),
    "CLIP_VIT_B_16" : (224,224),
    "CLIP_VIT_L_14" : (224,224),
    "CLIP_VIT_L_14_336" : (336,336),
    "CLIP_RN50" : (224,224),
    "CLIP_RN101" : (224,224),
    "CLIP_RN50x4" : (288,288),
    "CLIP_RN50x16" : (384,384),
    "CLIP_RN50x64" : (448,448),
    ## OpenCLIP Models
    "OPENCLIP_COCA_VIT_B_32" : (224, 224),
    "OPENCLIP_COCA_VIT_L_14" : (224, 224),
    "OPENCLIP_CONVNEXT_BASE" : (224, 224),
    "OPENCLIP_CONVNEXT_BASE_W" : (256, 256),
    "OPENCLIP_CONVNEXT_BASE_W_320" : (320, 320),
    "OPENCLIP_CONVNEXT_LARGE_D" : (256, 256),
    "OPENCLIP_CONVNEXT_LARGE_D_320" : (320, 320),
    "OPENCLIP_CONVNEXT_XXLARGE" : (256, 256),
    "OPENCLIP_EVA01_G_14" : (224, 224),
    "OPENCLIP_EVA01_G_14_PLUS" : (224, 224),
    "OPENCLIP_EVA02_B_16" : (224, 224),
    "OPENCLIP_EVA02_E_14" : (224, 224),
    "OPENCLIP_EVA02_E_14_PLUS" : (224, 224),
    "OPENCLIP_EVA02_L_14" : (224, 224),
    "OPENCLIP_EVA02_L_14_336" : (336, 336),
    "OPENCLIP_RN50" : (224, 224),
    "OPENCLIP_RN50_QUICKGELU" : (224, 224),
    "OPENCLIP_RN50X4" : (288, 288),
    "OPENCLIP_RN50X16" : (384, 384),
    "OPENCLIP_RN50X64" : (448, 448),
    "OPENCLIP_RN101" : (224, 224),
    "OPENCLIP_RN101_QUICKGELU" : (224, 224),
    "OPENCLIP_ROBERTA_VIT_B_32" : (224, 224),
    "OPENCLIP_VIT_B_16" : (224, 224),
    "OPENCLIP_VIT_B_16_PLUS_240" : (240, 240),
    "OPENCLIP_VIT_B_32" : (224, 224),
    "OPENCLIP_VIT_B_32_QUICKGELU" : (224, 224),
    "OPENCLIP_VIT_BIGG_14" : (224, 224),
    "OPENCLIP_VIT_G_14" : (224, 224),
    "OPENCLIP_VIT_H_14" : (224, 224),
    "OPENCLIP_VIT_L_14" : (224, 224),
    "OPENCLIP_VIT_L_14_336" : (336, 336),
    "OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32" : (224, 224),
    "OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14" : (224, 224),
}


SupportedModel_to_ModelName = {
    ## OpenAI CLIP models
    "CLIP_VIT_B_32" : "ViT-B/32",
    "CLIP_VIT_B_16" : "ViT-B/16",
    "CLIP_VIT_L_14" : "ViT-L/14",
    "CLIP_VIT_L_14_336" : "ViT-L/14@336px",
    "CLIP_RN50" : "RN50",
    "CLIP_RN101" : "RN101",
    "CLIP_RN50x4" : "RN50x4",
    "CLIP_RN50x16" : "RN50x16",
    "CLIP_RN50x64" : "RN50x64",
    ## OpenCLIP models
    "OPENCLIP_COCA_VIT_B_32" : "coca_ViT-B-32",
    "OPENCLIP_COCA_VIT_L_14" : "coca_ViT-L-14",
    "OPENCLIP_CONVNEXT_BASE" : "convnext_base",
    "OPENCLIP_CONVNEXT_BASE_W" : "convnext_base_w",
    "OPENCLIP_CONVNEXT_BASE_W_320" : "convnext_base_w_320",
    "OPENCLIP_CONVNEXT_LARGE_D" : "convnext_large_d",
    "OPENCLIP_CONVNEXT_LARGE_D_320" : "convnext_large_d_320",
    "OPENCLIP_CONVNEXT_XXLARGE" : "convnext_xxlarge",
    "OPENCLIP_EVA01_G_14" : "EVA01-g-14",
    "OPENCLIP_EVA01_G_14_PLUS" : "EVA01-g-14-plus",
    "OPENCLIP_EVA02_B_16" : "EVA02-B-16",
    "OPENCLIP_EVA02_E_14" : "EVA02-E-14",
    "OPENCLIP_EVA02_E_14_PLUS" : "EVA02-E-14-plus",
    "OPENCLIP_EVA02_L_14" : "EVA02-L-14",
    "OPENCLIP_EVA02_L_14_336" : "EVA02-L-14-336",
    "OPENCLIP_RN50" : "RN50",
    "OPENCLIP_RN50_QUICKGELU" : "RN50-quickgelu",
    "OPENCLIP_RN50X4" : "RN50x4",
    "OPENCLIP_RN50X16" : "RN50x16",
    "OPENCLIP_RN50X64" : "RN50x64",
    "OPENCLIP_RN101" : "RN101",
    "OPENCLIP_RN101_QUICKGELU" : "RN101-quickgelu",
    "OPENCLIP_ROBERTA_VIT_B_32" : "roberta-ViT-B-32",
    "OPENCLIP_VIT_B_16" : "ViT-B-16",
    "OPENCLIP_VIT_B_16_PLUS_240" : "ViT-B-16-plus-240",
    "OPENCLIP_VIT_B_32" : "ViT-B-32",
    "OPENCLIP_VIT_B_32_QUICKGELU" : "ViT-B-32-quickgelu",
    "OPENCLIP_VIT_BIGG_14" : "ViT-bigG-14",
    "OPENCLIP_VIT_G_14" : "ViT-g-14",
    "OPENCLIP_VIT_H_14" : "ViT-H-14",
    "OPENCLIP_VIT_L_14" : "ViT-L-14",
    "OPENCLIP_VIT_L_14_336" : "ViT-L-14-336",
    "OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32" : "xlm-roberta-base-ViT-B-32",
    "OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14" : "xlm-roberta-large-ViT-H-14",
}


SUPPORTED_VIDEO_FORMATS = ['.mp4']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, 'out-images')
OUT_VIDEOS_PATH = os.path.join(DATA_DIR_PATH, 'out-videos')
OUT_GIF_PATH = os.path.join(OUT_VIDEOS_PATH, 'GIFS')

# Make sure these exist as the rest of the code relies on it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)
os.makedirs(OUT_GIF_PATH, exist_ok=True)




