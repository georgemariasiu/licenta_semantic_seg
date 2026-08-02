import torch

NUM_CLASSES = 8

IMAGE_TEST = "data/test/uavid_test_base"
IMAGE_TRAIN = "data/train/uavid_train_base"
IMAGE_VAL = "data/val/uavid_val_base"

MASK_TRAIN = "data/train/uavid_train_mask"
MASK_VAL = "data/val/uavid_val_mask"

IMAGE_SIZE = (518, 518)
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")