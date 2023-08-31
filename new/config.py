import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/w223u672/AI_Projects/Pix2Pix_Revisited/data/trainingset/Impressionism"
VAL_DIR = "/home/w223u672/AI_Projects/Pix2Pix_Revisited/data/val"
LEARNING_RATE = 1.5e-4
BATCH_SIZE =  32
NUM_WORKERS = 2
IMAGE_SIZE = 1024
DATA_SIZE = 1024
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500 # 500 # 375
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "pos_disc.pth.tar"
CHECKPOINT_GEN = "pos_gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE), A.ColorJitter(p=0.2),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
