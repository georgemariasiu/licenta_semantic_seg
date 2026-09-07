import torch

NUM_CLASSES = 8

IMAGE_TEST = "data/test/uavid_test_base"
IMAGE_TRAIN = "data/train/uavid_train_base"
IMAGE_VAL = "data/val/uavid_val_base"

MASK_TRAIN = "data/train/uavid_train_mask"
MASK_VAL = "data/val/uavid_val_mask"

MODEL = "deeplabv3plus"
# MODEL = "dinov2linear"
# MODEL = "dinov2decoder"

NUM_EPOCHS = 30
PERSPECTIVE_SAFE = True

BATCH_SIZE = 4
WEIGHT_DECAY = 1e-4
WEIGHT_SCHEME = "log"

if MODEL == "deeplabv3plus":
    IMAGE_SIZE = (512, 512)
    LEARNING_RATE = 1e-3
elif MODEL in ("dinov2linear", "dinov2decoder"):
    IMAGE_SIZE = (518, 518)
    LEARNING_RATE = 1e-4
else:
    raise ValueError(f"Unknown model: {MODEL}")

RUN_NAME = (
    f"{MODEL}"
    f"_sz{IMAGE_SIZE[0]}"
    f"_lr{LEARNING_RATE}"
    f"_bs{BATCH_SIZE}"
    f"_ep{NUM_EPOCHS}"
    f"_ps{int(PERSPECTIVE_SAFE)}"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")