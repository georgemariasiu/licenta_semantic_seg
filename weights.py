import cv2
import numpy as np
import os
from dataset import UAVID_CLASS_MAP

def compute_class_weights(mask_dir, num_classes):

    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for filename in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, filename))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        R = mask[:, :, 0]
        G = mask[:, :, 1]
        B = mask[:, :, 2]

        for (r, g, b), cls in UAVID_CLASS_MAP.items():
            matches = (R == r) & (G == g) & (B == b)
            pixel_counts[cls] += matches.sum()

    frequencies = pixel_counts / pixel_counts.sum()
    weights = 1.0 / (frequencies + 1e-6)
    weights = weights / weights.mean()

    return weights