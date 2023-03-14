from augment import augmented_crop, correspondences

# import required libraries
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from threading import Thread
import vision_transformer as vits
import utils 
from vision_transformer import DINOHead

teacher = vits.__dict__['vit_tiny'](patch_size=16)
teacher = utils.MultiCropWrapper(
    teacher,
    DINOHead(192, 65536, False),
)

global_scale= 224
local_scale= 96

# Read image
image = Image.open('C://Users//alija/Desktop/test.png').convert('RGB')
  
# create an transform for crop the image
flip_and_color_jitter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.5),
])
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
my_transform = transforms.Compose([transforms.RandomResizedCrop(size=(96), scale=(0.1,0.15)),
                                ]) #flip_and_color_jitter, normalize

# first global crop
global_transfo1 = transforms.Compose([
    transforms.RandomResizedCrop(global_scale, scale=(0.4, 1.), interpolation=Image.Resampling.BICUBIC),
    flip_and_color_jitter,
    utils.GaussianBlur(1.0),
    normalize,
])

#second global crop
global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_scale, scale=(0.4, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

# first local crop
local_transfo = transforms.Compose([
    transforms.RandomResizedCrop(local_scale, scale=(0.05, 0.4), interpolation=Image.Resampling.BICUBIC),
    flip_and_color_jitter,
    utils.GaussianBlur(p=0.5),
    normalize,
])
############################################################################################################
patch_size = 16

#Global:
global1 = augmented_crop(global_transfo1, image, patch_size=patch_size, global_scale=global_scale, local_scale=local_scale)
global2 = augmented_crop(global_transfo2, image, patch_size=patch_size, global_scale=global_scale, local_scale=local_scale)

#Local:
local1 = augmented_crop(local_transfo, image, patch_size=patch_size, global_scale=global_scale, local_scale=local_scale)
local2 = augmented_crop(local_transfo, image, patch_size=patch_size, global_scale=global_scale, local_scale=local_scale)

# c1 = correspondences(global1, global2)
# correspondences(global1, local1, show_patches=True)
# correspondences(global2, local2)
c1 = correspondences(local1, local2, show_patches=True)
print(c1.selected_crop1_patches)
print(c1.selected_crop2_patches)
