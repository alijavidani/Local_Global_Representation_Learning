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
############################################################################################################

# def find_intersection(Crop1, Crop2):
#     top_intersection = max(Crop1.top, Crop2.top)
#     bottom_intersection = min(Crop1.bottom, Crop2.bottom)
#     left_intersection = max(Crop1.left, Crop2.left)
#     right_intersection = min(Crop1.right, Crop2.right)

#     if top_intersection >= bottom_intersection or left_intersection >= right_intersection:
#         print("no intersection")
#         intersection_coordinates = Coordinates(None, None, None, None)
#     else:
#         intersection_coordinates = Coordinates(top_intersection, bottom_intersection, left_intersection, right_intersection)
#     return intersection_coordinates


# def find_intersection_ids_in_crops(intersection_coordinates, crop_coordinates, local, flip):
#     patches = np.arange(1, self.max_limit**2 + 1).reshape([self.max_limit, self.max_limit])

#     width_ratio = (intersection_coordinates.right - intersection_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
#     height_ratio = (intersection_coordinates.bottom - intersection_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

#     width_quote = round(self.max_limit * width_ratio)
#     height_quote = round(self.max_limit * height_ratio)

#     start_pixel_width_ratio = (intersection_coordinates.left - crop_coordinates.left)/(crop_coordinates.right - crop_coordinates.left)
#     start_pixel_height_ratio = (intersection_coordinates.top - crop_coordinates.top)/(crop_coordinates.bottom - crop_coordinates.top)

#     start_patch_width = int(round(max_limit * start_pixel_width_ratio))
#     start_patch_height = int(round(max_limit * start_pixel_height_ratio))

#     end_patch_width = start_patch_width + width_quote
#     end_patch_height = start_patch_height + height_quote

#     if self.flip:
#         start_patch_width, end_patch_width = end_patch_width, start_patch_width
#         start_patch_width = max_limit - start_patch_width
#         end_patch_width = max_limit - end_patch_width

#     patches_view = patches[start_patch_height:end_patch_height, start_patch_width:end_patch_width]
#     if self.flip:
#         patches_view = np.fliplr(patches_view)

#     return patches_view


# def map_global2local(global_patches, local_patches):
#     global_columns = global_patches.shape[1]
#     global_rows = global_patches.shape[0]
#     local_columns = local_patches.shape[1]
#     local_rows = local_patches.shape[0]

#     if global_columns >= local_columns:
#         map_columns = [int(np.floor(i)) for i in np.linspace(0, global_columns-1, local_columns)]
#         if global_rows >= local_rows:
#             map_rows = [int(np.floor(i)) for i in np.linspace(0, global_rows-1, local_rows)]
#             selected_global_patches = global_patches[map_rows, :][:, map_columns]
#             selected_local_patches = local_patches
#         else:
#             map_rows = [int(np.floor(i)) for i in np.linspace(0, local_rows-1, global_rows)]
#             selected_global_patches = global_patches[:, map_columns]
#             selected_local_patches = local_patches[map_rows, :]

#     else:
#         map_columns = [int(np.floor(i)) for i in np.linspace(0, local_columns-1, global_columns)]
#         if global_rows >= local_rows:
#             map_rows = [int(np.floor(i)) for i in np.linspace(0, global_rows-1, local_rows)]
#             selected_global_patches = global_patches[map_rows, :]
#             selected_local_patches = local_patches[:, map_columns]
#         else:
#             map_rows = [int(np.floor(i)) for i in np.linspace(0, local_rows-1, global_rows)]
#             selected_global_patches = global_patches
#             selected_local_patches = local_patches[map_rows, :][:, map_columns]

#     return selected_global_patches, selected_local_patches



def display(im):
    im.show()

class Coordinates:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

class augmented_crop():
    def __init__(self, transformation, image):
        self.transformation = transformation
        self.step_size = 16

        self.original_image = image
        self.original_height = image.height
        self.original_width = image.width

        self.number_of_patches_per_row = self.original_width//self.step_size
        self.number_of_patches_per_column = self.original_height//self.step_size

        self.crop,[crop_properties, flip_and_color_jitter_returns] = transformation(image)
        print(flip_and_color_jitter_returns)
        
        self.is_local()
        print(self.local)

        self.find_coordinates(crop_properties)
        print(self.crop_coordinates.bottom)
        
        self.flip = flip_and_color_jitter_returns[0]
        print(self.flip)

        # self.find_intersection_ids_in_original_image()
        # print(self.indices_in_original_image)

    def is_local(self):
        if self.crop.size == (224,224):
            self.side_length = 224
            self.local = False

        elif self.crop.size == (96,96):
            self.side_length = 96
            self.local = True
        
        self.patches_per_side = self.side_length // self.step_size

    def draw_patches(self):
        self.crop_with_patches = self.crop.copy()
        shape1 = [(0, 0), (0, self.side_length)]
        shape2 = [(0, 0), (self.side_length, 0)]

        img1 = ImageDraw.Draw(self.crop_with_patches)
        for i in range(16, self.side_length,16):
            shape1[0] = (i, 0)
            shape1[1] = (i, self.side_length)
            shape2[0] = (0, i)
            shape2[1] = (self.side_length, i)
            img1.line(shape1, fill ="black", width = 2)
            img1.line(shape2, fill ="black", width = 2)
        return self.crop_with_patches

    def find_coordinates(self, crop_properties):
        i, j, h, w = crop_properties
        top = i
        bottom = i+h
        left = j
        right = j+w
        self.crop_coordinates = Coordinates(top, bottom, left, right)

    def find_intersection_ids_in_original_image(self):
        indices = []
        for i in range(self.crop_coordinates.top, self.crop_coordinates.bottom, self.step_size):
            for j in range(self.crop_coordinates.left, self.crop_coordinates.right, self.step_size):
                index_number = int((i/self.step_size) * self.number_of_patches_per_row + (j/self.step_size) + 1)
                indices.append(index_number)
        self.indices_in_original_image = indices

class correspondences():
    def __init__(self, augmented_crop1, augmented_crop2):
        self.augmented_crop1 = augmented_crop1
        self.augmented_crop2 = augmented_crop2

        self.find_intersection(augmented_crop1, augmented_crop2)
        print(f"top_intersection: {self.intersection_coordinates.top}, bottom_intersection: {self.intersection_coordinates.bottom}, left_intersection: {self.intersection_coordinates.left}, right_intersection: {self.intersection_coordinates.right}")

        self.crop1_patches = self.find_intersection_ids_in_crops(self.augmented_crop1)
        self.crop2_patches = self.find_intersection_ids_in_crops(self.augmented_crop2)
        print(self.crop1_patches)
        print(self.crop2_patches)

        self.selected_crop1_patches, self.selected_crop2_patches = self.map_global2local(self.crop1_patches, self.crop2_patches)
        print(self.selected_crop1_patches)
        print(self.selected_crop2_patches)

        self.show_patches()

    def find_intersection(self, augmented_crop1, augmented_crop2):
        top_intersection = max(augmented_crop1.crop_coordinates.top, augmented_crop2.crop_coordinates.top)
        bottom_intersection = min(augmented_crop1.crop_coordinates.bottom, augmented_crop2.crop_coordinates.bottom)
        left_intersection = max(augmented_crop1.crop_coordinates.left, augmented_crop2.crop_coordinates.left)
        right_intersection = min(augmented_crop1.crop_coordinates.right, augmented_crop2.crop_coordinates.right)

        if top_intersection >= bottom_intersection or left_intersection >= right_intersection:
            print("no intersection")
            self.intersection_coordinates = Coordinates(None, None, None, None)
        else:
            self.intersection_coordinates = Coordinates(top_intersection, bottom_intersection, left_intersection, right_intersection)


    def find_intersection_ids_in_crops(self, augmented_crop):
        patches = np.arange(1, augmented_crop.patches_per_side**2 + 1).reshape([augmented_crop.patches_per_side, augmented_crop.patches_per_side])

        width_ratio = (self.intersection_coordinates.right - self.intersection_coordinates.left)/(augmented_crop.crop_coordinates.right - augmented_crop.crop_coordinates.left)
        height_ratio = (self.intersection_coordinates.bottom - self.intersection_coordinates.top)/(augmented_crop.crop_coordinates.bottom - augmented_crop.crop_coordinates.top)

        width_quote = round(augmented_crop.patches_per_side * width_ratio)
        height_quote = round(augmented_crop.patches_per_side * height_ratio)

        start_pixel_width_ratio = (self.intersection_coordinates.left - augmented_crop.crop_coordinates.left)/(augmented_crop.crop_coordinates.right - augmented_crop.crop_coordinates.left)
        start_pixel_height_ratio = (self.intersection_coordinates.top - augmented_crop.crop_coordinates.top)/(augmented_crop.crop_coordinates.bottom - augmented_crop.crop_coordinates.top)

        start_patch_width = int(round(augmented_crop.patches_per_side * start_pixel_width_ratio))
        start_patch_height = int(round(augmented_crop.patches_per_side * start_pixel_height_ratio))

        end_patch_width = start_patch_width + width_quote
        end_patch_height = start_patch_height + height_quote

        if augmented_crop.flip:
            start_patch_width, end_patch_width = end_patch_width, start_patch_width
            start_patch_width = augmented_crop.patches_per_side - start_patch_width
            end_patch_width = augmented_crop.patches_per_side - end_patch_width

        patches_view = patches[start_patch_height:end_patch_height, start_patch_width:end_patch_width]
        if augmented_crop.flip:
            patches_view = np.fliplr(patches_view)

        return patches_view


    def map_global2local(self, global_patches, local_patches):
        global_rows, global_columns = global_patches.shape
        local_rows, local_columns = local_patches.shape

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


    def show_patches(self):
    # if show_patches == True:
        intersection_crop = self.augmented_crop1.original_image.crop((self.intersection_coordinates.left, self.intersection_coordinates.top, self.intersection_coordinates.right, self.intersection_coordinates.bottom))
        crop1 = self.augmented_crop1.draw_patches()
        crop2 = self.augmented_crop2.draw_patches()

        t1=Thread(target=display,args=(crop1,))
        t1.start()
        t2=Thread(target=display,args=(crop2,))
        t2.start()
        t3=Thread(target=display,args=(intersection_crop,))
        t3.start()

# def patch_correspondences(crop1, crop2, crop1_coordinates, crop2_coordinates, flip1, flip2, show_patches=False):
#     local1 = is_local(crop1)
#     local2 = is_local(crop2)

#     intersection_coordinates = find_intersection(crop1_coordinates, crop2_coordinates)
#     print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")

#     crop1_patches = find_intersection_ids_in_crops(intersection_coordinates, crop1_coordinates, local=local1, flip = local_flip)
#     crop2_patches = find_intersection_ids_in_crops(intersection_coordinates, crop2_coordinates, local=local2, flip = local_flip2)
#     print(crop1_patches)
#     print(crop2_patches)

#     selected_crop1_patches, selected_crop2_patches = map_global2local(crop1_patches, crop2_patches)
#     print(selected_crop1_patches)
#     print(selected_crop2_patches)

#     if show_patches == True:
#         intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))
#         crop1 = draw_patches(crop1)
#         crop2 = draw_patches(crop2)

#         t1=Thread(target=display,args=(crop1,))
#         t1.start()
#         t2=Thread(target=display,args=(crop2,))
#         t2.start()
#         t3=Thread(target=display,args=(intersection_crop,))
#         t3.start()


#Augment images and find coordinates of the crops:
#Global1:

global1 = augmented_crop(global_transfo1, image)
global2 = augmented_crop(global_transfo2, image)
local1 = augmented_crop(local_transfo, image)
local2 = augmented_crop(local_transfo, image)

# correspondences(global1, global2)
correspondences(global1, local1)
# correspondences(global2, local2)
# correspondences(local1, local2)

# global_crop,[global_crop_properties, global_flip_and_color_jitter_returns] = global_transfo1(image)
# print(global_flip_and_color_jitter_returns)
# global_flip = global_flip_and_color_jitter_returns[0]
# global_crop_coordinates = find_coordinates(global_crop_properties)

# #Global2:
# global_crop2,[global_crop_properties2, global_flip_and_color_jitter_returns2] = global_transfo2(image)
# print(global_flip_and_color_jitter_returns2)
# global_flip2 = global_flip_and_color_jitter_returns2[0]
# global_crop2_coordinates = find_coordinates(global_crop_properties2)

# #Local1:
# local_crop,[local_crop_properties, local_flip_and_color_jitter_returns] = local_transfo(image)
# print(local_flip_and_color_jitter_returns)
# local_flip = local_flip_and_color_jitter_returns[0]
# local_crop_coordinates = find_coordinates(local_crop_properties)

# #Local2:
# local_crop2,[local_crop_properties2, local_flip_and_color_jitter_returns2] = local_transfo(image)
# print(local_flip_and_color_jitter_returns2)
# local_flip2 = local_flip_and_color_jitter_returns2[0]
# local_crop2_coordinates = find_coordinates(local_crop_properties2)

############################################################################################################


# patch_correspondences(local_crop, local_crop2, local_crop_coordinates, local_crop2_coordinates, local_flip, local_flip2, show_patches=True)
# patch_correspondences(local_crop, global_crop, local_crop_coordinates, global_crop_coordinates, local_flip, global_flip, show_patches=True)
# patch_correspondences(local_crop, global_crop2, local_crop_coordinates, global_crop2_coordinates, local_flip, global_flip2, show_patches=True)
# patch_correspondences(global_crop, global_crop2, global_crop_coordinates, global_crop2_coordinates, global_flip, global_flip2, show_patches=True)
############################################################################################################

# intersection_coordinates = find_intersection(Crop1=crop_global, Crop2=crop_local)
# print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")

# intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))

# # create line image
# shape1 = [(0, 0), (0,224)]
# shape2 = [(0, 0), (224,0)]

# img1 = ImageDraw.Draw(global_crop)
# for i in range(16,224,16):
#     shape1[0] = (i,0)
#     shape1[1] = (i,224)
#     shape2[0] = (0,i)
#     shape2[1] = (224,i)
#     img1.line(shape1, fill ="black", width = 2)
#     img1.line(shape2, fill ="black", width = 2)

# # global_crop.show()

# shape3 = [(0, 0), (0,96)]
# shape4 = [(0, 0), (96,0)]

# img2 = ImageDraw.Draw(local_crop)
# for i in range(16,96,16):
#     shape3[0] = (i,0)
#     shape3[1] = (i,96)
#     shape4[0] = (0,i)
#     shape4[1] = (96,i)
#     img2.line(shape3, fill ="black", width = 2)
#     img2.line(shape4, fill ="black", width = 2)

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
# #2 Globals:
# intersection_coordinates = find_intersection(Crop1=crop_global, Crop2=crop_global2)
# print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")
# # image_crop_intersection = image.crop((left_intersection, top_intersection, right_intersection, bottom_intersection))

# intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))

# # create line image
# shape1 = [(0, 0), (0,224)]
# shape2 = [(0, 0), (224,0)]

# img1 = ImageDraw.Draw(global_crop)
# for i in range(16,224,16):
#     shape1[0] = (i,0)
#     shape1[1] = (i,224)
#     shape2[0] = (0,i)
#     shape2[1] = (224,i)
#     img1.line(shape1, fill ="black", width = 2)
#     img1.line(shape2, fill ="black", width = 2)

# # global_crop.show()

# shape3 = [(0, 0), (0,96)]
# shape4 = [(0, 0), (96,0)]

# img2 = ImageDraw.Draw(global_crop2)
# for i in range(16,224,16):
#     shape3[0] = (i,0)
#     shape3[1] = (i,224)
#     shape4[0] = (0,i)
#     shape4[1] = (224,i)
#     img2.line(shape3, fill ="black", width = 2)
#     img2.line(shape4, fill ="black", width = 2)

# # local_crop.show()

# t1=Thread(target=display,args=(global_crop,))
# t1.start()
# t2=Thread(target=display,args=(global_crop2,))
# t2.start()
# t3=Thread(target=display,args=(intersection_crop,))
# t3.start()

# global_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_global, local=False, flip = global_flip)
# local_patches = find_intersection_ids_in_crops(intersection_coordinates, crop_global2, local=False, flip = global_flip2)
# print(global_patches)
# print(local_patches)

# selected_global_patches, selected_local_patches = map_global2local(global_patches, local_patches)
# print(selected_global_patches)
# print(selected_local_patches)
############################################################################################################
#2 Locals:
# intersection_coordinates = find_intersection(Crop1=crop_local, Crop2=crop_local2)
# print(f"top_intersection: {intersection_coordinates.top}, bottom_intersection: {intersection_coordinates.bottom}, left_intersection: {intersection_coordinates.left}, right_intersection: {intersection_coordinates.right}")
# # image_crop_intersection = image.crop((left_intersection, top_intersection, right_intersection, bottom_intersection))

# intersection_crop = image.crop((intersection_coordinates.left, intersection_coordinates.top, intersection_coordinates.right, intersection_coordinates.bottom))

# local_crop = draw_patches(local_crop)
# local_crop2 = draw_patches(local_crop2)


# t1=Thread(target=display,args=(local_crop,))
# t1.start()
# t2=Thread(target=display,args=(local_crop2,))
# t2.start()
# t3=Thread(target=display,args=(intersection_crop,))
# t3.start()

# local_patches1 = find_intersection_ids_in_crops(intersection_coordinates, crop_local, local=True, flip = local_flip)
# local_patches2 = find_intersection_ids_in_crops(intersection_coordinates, crop_local2, local=True, flip = local_flip2)
# print(local_patches1)
# print(local_patches2)

# selected_local_patches1, selected_local_patches2 = map_global2local(local_patches1, local_patches2)
# print(selected_local_patches1)
# print(selected_local_patches2)


