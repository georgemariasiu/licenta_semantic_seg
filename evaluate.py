import torch
import numpy as np
from dataset import IGNORE_INDEX

def evaluate_model(model, loader, device, num_classes):

    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images).argmax(dim=1)

            valid = masks != IGNORE_INDEX
            y = masks[valid].view(-1)
            y_hat = preds[valid].view(-1)

            linear = y * num_classes + y_hat
            counts = torch.bincount(linear, minlength=num_classes ** 2)
            confusion += counts.view(num_classes, num_classes).cpu().numpy()

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    union = tp + fp + fn
    iou_per_class = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    miou = iou_per_class[union > 0].mean()

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

    return miou, iou_per_class, precision, recall, confusion