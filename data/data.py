# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:19:58 2024

@author: tejas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from tqdm import tqdm
from random import shuffle
from PIL import Image
import torch

from torch import nn
import math
from glob import glob
from patchify import patchify
import sys
import shutil

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

""" Create a directory """

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


save_path = "D:\projects\segmentation of aerial imagery\data\org_data"

create_dir("D:\projects\segmentation of aerial imagery\data\org_data\image")
create_dir("D:\projects\segmentation of aerial imagery\data\org_data\mask")

root = 'D:\projects\segmentation of aerial imagery\data'



def load_data(path):
    X = sorted(glob(os.path.join(path, "Tile*", "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "Tile*", "masks", "*.png")))
    return (X, Y)


def plotim(image, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()


images, masks = load_data(root)
patch_size = 256
mask_dataset = []
image_dataset = []
for idx, (X, Y) in tqdm(enumerate(zip(images, masks)), total=len(images)):

    name = X.split("\\")[-1].split(".")[0]

    image = cv2.imread(X)
    # plotim(image,mask)

    SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
    SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
    image = Image.fromarray(image)
    image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    image = np.array(image)

    # Extract patches from each image
    # print("Now patchifying image:",X)
    patches_img = patchify(image, (patch_size, patch_size, 3),
                           step=patch_size)  # Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]

            # Use minmaxscaler instead of just dividing by 255.
            # single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

            # single_patch_img = (single_patch_img.astype('float32')) / 255.
            single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
            image_dataset.append(single_patch_img)

    mask = cv2.imread(Y)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
    SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
    mask = Image.fromarray(mask)
    mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
    # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    mask = np.array(mask)

    # print("Now patchifying mask:",Y)
    patches_mask = patchify(mask, (patch_size, patch_size, 3),
                            step=patch_size)  # Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
            single_patch_mask = single_patch_mask[0]  # Drop the extra unecessary dimension that patchify adds.
            mask_dataset.append(single_patch_mask)

index = 0
for i, m in tqdm(zip(image_dataset, mask_dataset)):
    #plotim(i,m)

    tmp_image_name = f"{name}_{index}.png"
    tmp_mask_name = f"{name}_{index}.png"

    image_path = os.path.join(save_path, "image", tmp_image_name)
    mask_path = os.path.join(save_path, "mask", tmp_mask_name)

    cv2.imwrite(image_path, i)
    cv2.imwrite(mask_path, m)

    index += 1










