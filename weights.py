import cv2
import json
import numpy as np
import os
from dataset import UAVID_CLASS_MAP

def compute_class_distribution(mask_dir, num_classes, size=None, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return np.array(cached["counts"], dtype=np.int64), cached["unmatched"]

    counts = np.zeros(num_classes, dtype=np.int64)
    unmatched = 0

    palette = np.array(list(UAVID_CLASS_MAP.keys()), dtype=np.uint8)
    indices = np.array(list(UAVID_CLASS_MAP.values()), dtype=np.int64)

    for filename in sorted(os.listdir(mask_dir)):
        mask = cv2.imread(os.path.join(mask_dir, filename))
        if mask is None:
            continue
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if size is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        total_pixels = mask.shape[0] * mask.shape[1]
        matched = 0

        for i, color in enumerate(palette):
            is_this_color = np.all(mask == color, axis=2)
            n = int(is_this_color.sum())
            counts[indices[i]] += n
            matched += n

        unmatched += total_pixels - matched

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"counts": counts.tolist(), "unmatched": int(unmatched)}, f)

    return counts, unmatched


def compute_class_weights(mask_dir, num_classes, size=None, scheme="log", beta=1.02, cache_path=None, verbose=True):

    counts, unmatched = compute_class_distribution(mask_dir, num_classes, size=size, cache_path=cache_path)

    total = counts.sum()
    if total == 0:
        raise ValueError(f"no palette-matching pixels found in {mask_dir}")
    freq = counts / total
    epsilon = 1e-6

    # raw inverse frequency
    if scheme == "inverse":
        w = 1.0 / (freq + epsilon)
    # compressed inverse frequency
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(freq + epsilon)
    # ENet-style, strongest compression
    elif scheme == "log":
        w = 1.0 / np.log(beta + freq)
    # median-frequency balancing
    elif scheme == "median":
        present = freq[freq > 0]
        w = np.median(present) / (freq + epsilon)
    elif scheme == "none":
        w = np.ones(num_classes)
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    w[counts == 0] = 0.0
    w = w / w[w > 0].mean()

    if verbose:
        if unmatched:
            pct = 100.0 * unmatched / (unmatched + total)
            print(f"[weights] WARNING {unmatched:,} pixels ({pct:.3f}%) matched no palette colour")
        print(f"[weights] scheme={scheme}  ratio max/min = {w[w > 0].max() / w[w > 0].min():.1f} : 1")
        for c in range(num_classes):
            print(f"[weights]   class {c}: share {100 * freq[c]:6.3f}%   weight {w[c]:7.3f}")

    return w.astype(np.float32), freq