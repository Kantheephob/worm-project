import torch

# weights path
CONVNEXT_BASE_PATH = '/project/lt200264-saiwat/pretrained_models/convnext/convnext_base.pth'
CONVNEXT_LARGE_PATH = '/project/lt200264-saiwat/pretrained_models/convnext/convnext_large.pth'
CONVNEXT_SMALL_PATH = '/project/lt200264-saiwat/pretrained_models/convnext/convnext_small.pth'
CONVNEXT_TINY_PATH = '/project/lt200264-saiwat/pretrained_models/convnext/convnext_tiny.pth'

EFFICIENTNET_V2_L_PATH = '/project/lt200264-saiwat/pretrained_models/efficientnet_v2/efficientnet_v2_l.pth'
EFFICIENTNET_V2_M_PATH = '/project/lt200264-saiwat/pretrained_models/efficientnet_v2/efficientnet_v2_m.pth'
EFFICIENTNET_V2_S_PATH = '/project/lt200264-saiwat/pretrained_models/efficientnet_v2/efficientnet_v2_s.pth'

RESNET101_PATH = '/project/lt200264-saiwat/pretrained_models/resnet/resnet101.pth'
RESNET152_PATH = '/project/lt200264-saiwat/pretrained_models/resnet/resnet152.pth'
RESNET18_PATH = '/project/lt200264-saiwat/pretrained_models/resnet/resnet18.pth'
RESNET34_PATH = '/project/lt200264-saiwat/pretrained_models/resnet/resnet34.pth'
RESNET50_PATH = '/project/lt200264-saiwat/pretrained_models/resnet/resnet50.pth'

# (video models)
# SWIN3D_B_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_b.pth'
# SWIN3D_S_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_s.pth'
# SWIN3D_T_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_t.pth'
SWIN_B_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_b.pth'
SWIN_S_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_s.pth'
SWIN_T_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_t.pth'
SWIN_V2_B_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_v2_b.pth'
SWIN_V2_S_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_v2_s.pth'
SWIN_V2_T_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin_v2_t.pth'

VIT_B_16_PATH = '/project/lt200264-saiwat/pretrained_models/vit/vit_b_16.pth'
VIT_B_32_PATH = '/project/lt200264-saiwat/pretrained_models/vit/vit_b_32.pth'
VIT_H_14_PATH = '/project/lt200264-saiwat/pretrained_models/vit/vit_h_14.pth'
VIT_L_16_PATH = '/project/lt200264-saiwat/pretrained_models/vit/vit_l_16.pth'
VIT_L_32_PATH = '/project/lt200264-saiwat/pretrained_models/vit/vit_l_32.pth'

YOLO26N_CLS_PATH = '/project/lt200264-saiwat/pretrained_models/Yolo/yolo26/yolo26n-cls.pt'
YOLO26S_CLS_PATH = '/project/lt200264-saiwat/pretrained_models/Yolo/yolo26/yolo26s-cls.pt'
YOLO26M_CLS_PATH = '/project/lt200264-saiwat/pretrained_models/Yolo/yolo26/yolo26m-cls.pt'
YOLO26L_CLS_PATH = '/project/lt200264-saiwat/pretrained_models/Yolo/yolo26/yolo26l-cls.pt'
YOLO26X_CLS_PATH = '/project/lt200264-saiwat/pretrained_models/Yolo/yolo26/yolo26x-cls.pt'

# experiments dir
OUTPUT_DIR = '/scratch/lt200264-saiwat/worm-project/worm-classification-model'

# experiment name
EXPERIMENT_NAME = 'model_benchmark'

# dataset path
CLS2 = '/project/lt200264-saiwat/WormProject/data/classify-v1/cls_2classes'
CLS5 = '/project/lt200264-saiwat/WormProject/data/classify-v1/cls_5classes'
CLS2_PAD0_BLACKOUT = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls2_pad0_blackout'
CLS2_PAD30_DIM0P3 = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls2_pad30_dim0p3'
CLS2_PAD30_RAW = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls2_pad30_raw'
CLS3_STAGE_PAD30 = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls3_stage_pad30'
CLS4_PAD30 = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls4_pad30'
CLS5_PAD30 = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v1/views/cls5_pad30'

# name map
DATASETS = {
    '2class': CLS2,
    '5class': CLS5,
    '2class_pad0_blackout': CLS2_PAD0_BLACKOUT,
    '2class_pad30_dim': CLS2_PAD30_DIM0P3,
    '2class_pad30_raw': CLS2_PAD30_RAW,
    '3class_stage_pad30': CLS3_STAGE_PAD30,
    '4class_pad30': CLS4_PAD30,
    '5class_pad30': CLS5_PAD30,
}

# 2 class calculate
GOOD_CLASS_NAMES = ['good', 'Perfect']

# seed
SEED = 42

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU
NUM_GPUS = 4

# dataloader
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

# train
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
START_FACTOR = 0.1
END_FACTOR = 1.0
ETA_MIN = 1e-5
USE_AMP = True
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 10