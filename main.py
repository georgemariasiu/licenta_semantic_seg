import evaluate, train, weights
from config import param_uavid
from dataset import IGNORE_INDEX
from dataset import UAVid
from dataset import UAVID_CLASS_MAP
from models import model_deeplabv3plus, model_dinov2_linear, model_dinov2_decoder

import cv2
import json
import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

UAVID_CLASS_NAMES = {
    "building",
    "road",
    "static car",
    "tree",
    "low vegetation",
    "human",
    "moving car",
    "background clutter"
}

BUILDERS = {
    "deeplabv3plus": model_deeplabv3plus.create_model,
    "dinov2_linear": model_dinov2_linear.create_model,
    "dinov2_decoder": model_dinov2_decoder.create_model
}

if __name__ == '__main__':

    checkpoint_dir = os.path.join("checkpoints", param_uavid.RUN_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = UAVid(
        image_dir=param_uavid.IMAGE_TRAIN,
        mask_dir=param_uavid.MASK_TRAIN,
        size=param_uavid.IMAGE_SIZE,
        augment=True,
        perspective_safe=param_uavid.PERSPECTIVE_SAFE,
    )
    val_dataset = UAVid(
        image_dir=param_uavid.IMAGE_VAL,
        mask_dir=param_uavid.MASK_VAL,
        size=param_uavid.IMAGE_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=param_uavid.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=param_uavid.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    ###

    model = BUILDERS[param_uavid.MODEL](param_uavid.NUM_CLASSES)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=param_uavid.LEARNING_RATE,
        weight_decay=param_uavid.WEIGHT_DECAY
    )

    class_weights, class_freq = weights.compute_class_weights(
        param_uavid.MASK_TRAIN,
        param_uavid.NUM_CLASSES,
        size=(param_uavid.IMAGE_SIZE[1], param_uavid.IMAGE_SIZE[0]),
        scheme=param_uavid.WEIGHT_SCHEME,
        cache_path=f"cache/train_dist_{param_uavid.IMAGE_SIZE[0]}.json"
    )
    class_weights_np = np.asarray(class_weights, dtype=np.float64)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(param_uavid.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )

    tstart = time.time()
    model, best_epoch, best_val_loss = train.train_model(
        model,
        train_loader,
        val_loader,
        param_uavid.DEVICE,
        optimizer,
        criterion,
        scheduler,
        param_uavid.RUN_NAME,
        param_uavid.NUM_EPOCHS,
        resume=None
    )
    tend = time.time() - tstart

    ###

    latest_miou, latest_iou, latest_prec, latest_rec, latest_cm = evaluate.evaluate_model(model, val_loader, param_uavid.DEVICE, param_uavid.NUM_CLASSES)

    print(f"Latest mIoU: {latest_miou}")
    for i, iou in enumerate(latest_iou):
        print(f"  Class {i}: {iou:.4f}")

    best_path = os.path.join(checkpoint_dir, "best.pth")
    model.load_state_dict(torch.load(best_path, map_location=param_uavid.DEVICE, weights_only=True))
    model = model.to(param_uavid.DEVICE)

    best_miou, best_iou, best_prec, best_rec, best_cm = evaluate.evaluate_model(model, val_loader, param_uavid.DEVICE, param_uavid.NUM_CLASSES)

    print(f"Best mIoU: {best_miou}")
    for i, iou in enumerate(best_iou):
        print(f"  Class {i}: {iou:.4f}")

    ###

    palette = np.zeros((param_uavid.NUM_CLASSES, 3), dtype=np.uint8)
    for color, idx in UAVID_CLASS_MAP.items():
        palette[idx] = color

    sample_idx = 0
    image_name = val_dataset.images[sample_idx]
    image_tensor, _ = val_dataset[sample_idx]
    image_tensor = image_tensor.unsqueeze(0).to(param_uavid.DEVICE)

    og = cv2.imread(os.path.join(param_uavid.IMAGE_VAL, image_name))
    og_h, og_w = og.shape[:2]

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred = cv2.resize(pred, (og_w, og_h), interpolation=cv2.INTER_NEAREST)
    colored = palette[pred]

    out_path = os.path.join(checkpoint_dir, "best_prediction.png")
    cv2.imwrite(out_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print(f"Saved prediction mask to {out_path}")

    # load the best-loss weights (in-memory model is the last epoch, not the best)
    # class index -> RGB color palette (inverse of UAVID_CLASS_MAP)
    # pick one validation image (first one); preprocess via the dataset so it
    # matches training exactly (augment=False on val_dataset)
    # original resolution, so the saved mask matches the source image dimensions
    # back to original size; nearest keeps it to valid class indices only
    # every pixel -> its class color

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
