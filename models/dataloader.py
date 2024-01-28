import os
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import torch.nn as nn

import os
import time
from glob import glob
class Aerial(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(Aerial, self).__init__()
        self.root = root
        self.transform = transform
        self.IMG_NAMES = sorted(glob(self.root + '/images/*.png'))
        self.color_codes = {
            "Building": "3C1098",
            "Land": "8429F6",
            "Road": "6EC1E4",
            "Vegetation": "FEDD3A",
            "Water": "E2A929",
            "Unlabeled": "9B9B9B"
        }

        self.BGR_classes = self.convert_hex_to_bgr(self.color_codes)
        self.bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

    def convert_hex_to_bgr(self, hex_codes):
        bgr_classes = {}
        for key, value in hex_codes.items():
            rgb = np.array(tuple(int(value[i:i+2], 16) for i in (0, 2, 4)))
            bgr = rgb[::-1]  # Convert RGB to BGR
            bgr_classes[key] = bgr.tolist()
        return bgr_classes

    def __getitem__(self, idx):
          img_path = self.IMG_NAMES[idx]
          mask_path = img_path.replace('image\\', 'mask\\')

          image = cv2.imread(img_path)
          mask = cv2.imread(mask_path)
          cls_mask = np.zeros(mask.shape)
          cls_mask[mask == self.BGR_classes['Water']] = self.bin_classes.index('Water')
          cls_mask[mask == self.BGR_classes['Land']] = self.bin_classes.index('Land')
          cls_mask[mask == self.BGR_classes['Road']] = self.bin_classes.index('Road')
          cls_mask[mask == self.BGR_classes['Building']] = self.bin_classes.index('Building')
          cls_mask[mask == self.BGR_classes['Vegetation']] = self.bin_classes.index('Vegetation')
          cls_mask[mask == self.BGR_classes['Unlabeled']] = self.bin_classes.index('Unlabeled')

          cls_mask = cls_mask[:,:,0]
          image = cv2.resize(image, (512,512))/255.0
          cls_mask = cv2.resize(cls_mask, (512,512))
          image = np.moveaxis(image, -1, 0)

          return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)
    def __len__(self):
          return len(self.IMG_NAMES)



datadir="root=r'D:\projects\segmentation of aerial imagery\data\org_data'"
print(Aerial(datadir))

