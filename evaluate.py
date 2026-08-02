import torch
import numpy as np

def evaluate_model(model, loader, device, num_classes):

    model.eval()
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for cls in range(num_classes):
                pred_c = (preds == cls)
                true_c = (masks == cls)
                intersection[cls] += (pred_c & true_c).sum().item()
                union[cls] += (pred_c | true_c).sum().item()

    iou_per_class = intersection / (union + 1e-6)
    miou = iou_per_class[union > 0].mean()
    return miou, iou_per_class