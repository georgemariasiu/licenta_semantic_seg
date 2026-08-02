import cv2
import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

# building, road, static car, tree, low vegetation, human, moving car, background clutter
UAVID_CLASS_MAP = {
    (128, 0, 0): 0,
    (128, 64, 128): 1,
    (192, 0, 192): 2,
    (0, 128, 0): 3,
    (128, 128, 0): 4,
    (64, 64, 0): 5,
    (64, 0, 128): 6,
    (0, 0, 0): 7
}

class UAVid(Dataset):

    def __init__(self, image_dir, mask_dir, size=(512, 512), augment=False):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):

        return len(self.images)

    def geometric_augmentations(self, image, mask):

        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        if random.random() < 0.5:
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(int(self.size[0] * 0.8), int(self.size[1] * 0.8)))
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)

        return image, mask

    def color_augmentations(self, image):

        if random.random() < 0.5:
            image = F.adjust_brightness(image, random.uniform(0.7, 1.3))

        if random.random() < 0.5:
            image = F.adjust_contrast(image, random.uniform(0.7, 1.3))

        if random.random() < 0.5:
            image = F.adjust_saturation(image, random.uniform(0.7, 1.3))

        if random.random() < 0.5:
            image = F.adjust_hue(image, random.uniform(-0.1, 0.1))

        return image

    def __getitem__(self, i):

        image = cv2.imread(self.image_dir + '/' + self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_dir + '/' + self.images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask_new = np.zeros(mask.shape[:2], dtype=np.uint8)

        R_channel = mask[:, :, 0]
        G_channel = mask[:, :, 1]
        B_channel = mask[:, :, 2]

        for color, j in UAVID_CLASS_MAP.items():
            r, g, b = color
            matches = ((R_channel == r) & (G_channel == g) & (B_channel == b))
            mask_new[matches] = j

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_new).long()

        if self.size is not None:
            image_tensor = F.resize(image_tensor, self.size, interpolation=F.InterpolationMode.BILINEAR)
            mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = F.resize(mask_tensor, self.size, interpolation=F.InterpolationMode.NEAREST)
            mask_tensor = mask_tensor.squeeze(0)

        if self.augment:
            mask_tensor = mask_tensor.unsqueeze(0)
            image_tensor, mask_tensor = self.geometric_augmentations(image_tensor, mask_tensor)
            mask_tensor = mask_tensor.squeeze(0)
            image_tensor = self.color_augmentations(image_tensor)

        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image_tensor, mask_tensor