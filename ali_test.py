# import required libraries
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from threading import Thread
import vision_transformer as vits
import utils 
from vision_transformer import DINOHead

class Coordinates:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

teacher = vits.__dict__['vit_tiny'](patch_size=16)
teacher = utils.MultiCropWrapper(
    teacher,
    DINOHead(192, 65536, False),
)

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
    transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.Resampling.BICUBIC),
    flip_and_color_jitter,
    utils.GaussianBlur(1.0),
    # normalize,
])

#second global crop
global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            # normalize,
        ])

# first local crop
local_transfo = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.05, 0.4), interpolation=Image.Resampling.BICUBIC),
    flip_and_color_jitter,
    utils.GaussianBlur(p=0.5),
    # normalize,
])

# print(global_transfo1(image))
global_crop,[global_crop_coordinates, global_flip_and_color_jitter_returns] = global_transfo1(image)
print(global_flip_and_color_jitter_returns)
global_flip = global_flip_and_color_jitter_returns[0]
i1, j1, h1, w1 = global_crop_coordinates
top1 = i1
bottom1 = i1+h1
left1 = j1
right1 = j1+w1
crop_global = Coordinates(top1, bottom1, left1, right1)

local_crop,[local_crop_coordinates, local_flip_and_color_jitter_returns] = local_transfo(image)
print(local_flip_and_color_jitter_returns)
local_flip = local_flip_and_color_jitter_returns[0]
i2, j2, h2, w2 = local_crop_coordinates
top2 = i2
bottom2 = i2+h2
left2 = j2
right2 = j2+w2
crop_local = Coordinates(top2, bottom2, left2, right2)

global_crop2,[global_crop_coordinates2, global_flip_and_color_jitter_returns2] = global_transfo2(image)
print(global_flip_and_color_jitter_returns2)
global_flip2 = global_flip_and_color_jitter_returns2[0]
i3, j3, h3, w3 = global_crop_coordinates2
top3 = i3
bottom3 = i3+h3
left3 = j3
right3 = j3+w3
crop_global2 = Coordinates(top3, bottom3, left3, right3)


# print(np.arange(top1,bottom1), np.arange(left1,right1))
# print(np.arange(top2,bottom2),np.arange(left2,right2))

# print(f"top1: {top1}, bottom1: {bottom1}, left1: {left1}, right1: {right1}")
# print(f"top2: {top2}, bottom2: {bottom2}, left2: {left2}, right2: {right2}")


def find_intersection(Crop1, Crop2):
    top_intersection = max(Crop1.top, Crop2.top)
    bottom_intersection = min(Crop1.bottom, Crop2.bottom)
    left_intersection = max(Crop1.left, Crop2.left)
    right_intersection = min(Crop1.right, Crop2.right)

    if top_intersection >= bottom_intersection or left_intersection >= right_intersection:
        intersection_coordinates = Coordinates(None, None, None, None)
    else:
        intersection_coordinates = Coordinates(top_intersection, bottom_intersection, left_intersection, right_intersection)
    return intersection_coordinates


def find_intersection_ids_in_original_image(Coordinates, original_height, original_width, step_size = 16):
    number_of_patches_per_row = original_width//step_size
    indices = []
    for i in range(Coordinates.top, Coordinates.bottom, step_size):
        for j in range(Coordinates.left, Coordinates.right, step_size):
            index_number = int((i/step_size)*number_of_patches_per_row + (j/step_size) + 1)
            indices.append(index_number)
    return indices


# def find_intersection_ids_in_crops_old(intersection_coordinates, crop_coordinates, local):
#     width_ratio = (intersection_coordinates.right - intersection_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
#     height_ratio = (intersection_coordinates.bottom - intersection_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

#     width_quote = round(6 * width_ratio) if local else round(14 * width_ratio)
#     height_quote = round(6 * height_ratio) if local else round(14 * height_ratio)

#     start_pixel_width_ratio = (intersection_coordinates.left - crop_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
#     start_pixel_height_ratio = (intersection_coordinates.top - crop_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

#     start_patch_width = int(round(6 * start_pixel_width_ratio)) if local else int(round(14 * start_pixel_width_ratio))
#     start_patch_height = int(round(6 * start_pixel_height_ratio)) if local else int(round(14 * start_pixel_height_ratio))

#     end_patch_width = start_patch_width + width_quote
#     end_patch_height = start_patch_height + height_quote

#     if local:
#         patches_local = np.arange(1,37).reshape([6,6])
#         patches_local_view = patches_local[start_patch_height:end_patch_height, start_patch_width:end_patch_width]
#         return patches_local_view
#     else:
#         patches_global = np.arange(1,197).reshape([14,14])
#         patches_global_view = patches_global[start_patch_height:end_patch_height, start_patch_width:end_patch_width]
#         return patches_global_view

def find_intersection_ids_in_crops(intersection_coordinates, crop_coordinates, local, flip):
    if local:
        max_limit = 6
    else:
        max_limit = 14
    patches = np.arange(1, max_limit**2 + 1).reshape([max_limit, max_limit])

    width_ratio = (intersection_coordinates.right - intersection_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
    height_ratio = (intersection_coordinates.bottom - intersection_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

    width_quote = round(max_limit * width_ratio)
    height_quote = round(max_limit * height_ratio)

    start_pixel_width_ratio = (intersection_coordinates.left - crop_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
    start_pixel_height_ratio = (intersection_coordinates.top - crop_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

    start_patch_width = int(round(max_limit * start_pixel_width_ratio))
    start_patch_height = int(round(max_limit * start_pixel_height_ratio))

    end_patch_width = start_patch_width + width_quote
    end_patch_height = start_patch_height + height_quote

    if flip:
        start_patch_width, end_patch_width = end_patch_width, start_patch_width
        start_patch_width = max_limit - start_patch_width
        end_patch_width = max_limit - end_patch_width

    patches_view = patches[start_patch_height:end_patch_height, start_patch_width:end_patch_width]
    if flip:
        patches_view = np.fliplr(patches_view)

    return patches_view


def map_global2local(global_patches, local_patches):
    global_columns = global_patches.shape[1]
    global_rows = global_patches.shape[0]
    local_columns = local_patches.shape[1]
    local_rows = local_patches.shape[0]

    if global_columns >= local_columns:
        map_columns = [int(np.floor(i)) for i in np.linspace(0, global_columns-1, local_columns)]
        if global_rows >= local_rows:
            map_rows = [int(np.floor(i)) for i in np.linspace(0, global_rows-1, local_rows)]
            selected_global_patches = global_patches[map_rows, :][:, map_columns]
            selected_local_patches = local_patches
        else:
            map_rows = [int(np.floor(i)) for i in np.linspace(0, local_rows-1, global_rows)]
            selected_global_patches = global_patches[:, map_columns]
            selected_local_patches = local_patches[map_rows, :]

    else:
        map_columns = [int(np.floor(i)) for i in np.linspace(0, local_columns-1, global_columns)]
        if global_rows >= local_rows:
            map_rows = [int(np.floor(i)) for i in np.linspace(0, global_rows-1, local_rows)]
            selected_global_patches = global_patches[map_rows, :]
            selected_local_patches = local_patches[:, map_columns]
        else:
            map_rows = [int(np.floor(i)) for i in np.linspace(0, local_rows-1, global_rows)]
            selected_global_patches = global_patches
            selected_local_patches = local_patches[map_rows, :][:, map_columns]

    return selected_global_patches, selected_local_patches


def display(im):
    im.show()

intersection_coordinates = find_intersection(Crop1=crop_global, Crop2=crop_local)
print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")
# image_crop_intersection = image.crop((left_intersection, top_intersection, right_intersection, bottom_intersection))

intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))

# create line image
shape1 = [(0, 0), (0,224)]
shape2 = [(0, 0), (224,0)]

img1 = ImageDraw.Draw(global_crop)
for i in range(16,224,16):
    shape1[0] = (i,0)
    shape1[1] = (i,224)
    shape2[0] = (0,i)
    shape2[1] = (224,i)
    img1.line(shape1, fill ="black", width = 2)
    img1.line(shape2, fill ="black", width = 2)

# global_crop.show()

shape3 = [(0, 0), (0,96)]
shape4 = [(0, 0), (96,0)]

img2 = ImageDraw.Draw(local_crop)
for i in range(16,96,16):
    shape3[0] = (i,0)
    shape3[1] = (i,96)
    shape4[0] = (0,i)
    shape4[1] = (96,i)
    img2.line(shape3, fill ="black", width = 2)
    img2.line(shape4, fill ="black", width = 2)

# local_crop.show()

# t1=Thread(target=display,args=(global_crop,))
# t1.start()
# t2=Thread(target=display,args=(local_crop,))
# t2.start()
# t3=Thread(target=display,args=(intersection_crop,))
# t3.start()

# global_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_global, local=False, flip = global_flip)
# local_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_local, local=True, flip = local_flip)
# print(global_patches)
# print(local_patches)

# selected_global_patches, selected_local_patches = map_global2local(global_patches, local_patches)
# print(selected_global_patches)
# print(selected_local_patches)

############################################################################################################
#2 Globals:
intersection_coordinates = find_intersection(Crop1=crop_global, Crop2=crop_global2)
print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")
# image_crop_intersection = image.crop((left_intersection, top_intersection, right_intersection, bottom_intersection))

intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))

# create line image
shape1 = [(0, 0), (0,224)]
shape2 = [(0, 0), (224,0)]

img1 = ImageDraw.Draw(global_crop)
for i in range(16,224,16):
    shape1[0] = (i,0)
    shape1[1] = (i,224)
    shape2[0] = (0,i)
    shape2[1] = (224,i)
    img1.line(shape1, fill ="black", width = 2)
    img1.line(shape2, fill ="black", width = 2)

# global_crop.show()

shape3 = [(0, 0), (0,96)]
shape4 = [(0, 0), (96,0)]

img2 = ImageDraw.Draw(global_crop2)
for i in range(16,224,16):
    shape3[0] = (i,0)
    shape3[1] = (i,224)
    shape4[0] = (0,i)
    shape4[1] = (224,i)
    img2.line(shape3, fill ="black", width = 2)
    img2.line(shape4, fill ="black", width = 2)

# local_crop.show()

t1=Thread(target=display,args=(global_crop,))
t1.start()
t2=Thread(target=display,args=(global_crop2,))
t2.start()
t3=Thread(target=display,args=(intersection_crop,))
t3.start()

global_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_global, local=False, flip = global_flip)
local_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_global2, local=False, flip = global_flip2)
print(global_patches)
print(local_patches)

selected_global_patches, selected_local_patches = map_global2local(global_patches, local_patches)
print(selected_global_patches)
print(selected_local_patches)
############################################################################################################
# Mesale ALAKI!

#intersection:
top_intersection = 128
bottom_intersection = 256
left_intersection = 64
right_intersection = 160
intersection_coordinates = Coordinates(top_intersection, bottom_intersection, left_intersection, right_intersection)

#image_local:
top_image_local = 64
bottom_image_local = 256
left_image_local = 16
right_image_local = 160
local_coordinates = Coordinates(top_image_local, bottom_image_local, left_image_local, right_image_local)

#image_global:
top_image_global = 128
bottom_image_global = 320
left_image_global = 64
right_image_global = 256
global_coordinates = Coordinates(top_image_global, bottom_image_global, left_image_global, right_image_global)

# original_height = 320
# original_width = 256

# indices = find_ids(32, 64, 0, 32, 64, 32)
# indices = find_ids(128, 256, 64, 160, 320, 256)
# indices = find_ids(64, 256, 16, 160, 320, 256)
# print(indices)


# original_width = 160 - 16
# original_height = 256 - 64
# step_size = 16
# number_of_patches_per_row = original_width//step_size
# number_of_patches_per_column = original_height//step_size
# image1_patches = np.array(indices).reshape([number_of_patches_per_column, number_of_patches_per_row])
# print(image1_patches)

# global_patches = find_intersection_ids_in_crops(intersection_coordinates, global_coordinates, local=False)
# local_patches = find_intersection_ids_in_crops (intersection_coordinates, local_coordinates, local=True)
# print(global_patches)
# print(local_patches)

# selected_global_patches = map_global2local(global_patches, local_patches)
# print(selected_global_patches)

