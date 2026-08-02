import segmentation_models_pytorch as smp

def create_model(num_classes):

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )

    return model