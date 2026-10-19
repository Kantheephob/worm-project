import torch

# # weights path
# CONVNEXT_BASE_PATH = '/home/ml_trainer/worm-project/pt-models/convnext/convnext_base.pth'
# CONVNEXT_LARGE_PATH = '/home/ml_trainer/worm-project/pt-models/convnext/convnext_large.pth'
# CONVNEXT_SMALL_PATH = '/home/ml_trainer/worm-project/pt-models/convnext/convnext_small.pth'
# CONVNEXT_TINY_PATH = '/home/ml_trainer/worm-project/pt-models/convnext/convnext_tiny.pth'

# EFFICIENTNET_V2_L_PATH = '/home/ml_trainer/worm-project/pt-models/efficientnet_v2/efficientnet_v2_l.pth'
# EFFICIENTNET_V2_M_PATH = '/home/ml_trainer/worm-project/pt-models/efficientnet_v2/efficientnet_v2_m.pth'
# EFFICIENTNET_V2_S_PATH = '/home/ml_trainer/worm-project/pt-models/efficientnet_v2/efficientnet_v2_s.pth'

# RESNET101_PATH = '/home/ml_trainer/worm-project/pt-models/resnet/resnet101.pth'
# RESNET152_PATH = '/home/ml_trainer/worm-project/pt-models/resnet/resnet152.pth'
# RESNET18_PATH = '/home/ml_trainer/worm-project/pt-models/resnet/resnet18.pth'
# RESNET34_PATH = '/home/ml_trainer/worm-project/pt-models/resnet/resnet34.pth'
# RESNET50_PATH = '/home/ml_trainer/worm-project/pt-models/resnet/resnet50.pth'

# # (video models)
# # SWIN3D_B_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_b.pth'
# # SWIN3D_S_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_s.pth'
# # SWIN3D_T_PATH = '/project/lt200264-saiwat/pretrained_models/swin/swin3d_t.pth'
# SWIN_B_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_b.pth'
# SWIN_S_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_s.pth'
# SWIN_T_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_t.pth'
# SWIN_V2_B_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_v2_b.pth'
# SWIN_V2_S_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_v2_s.pth'
# SWIN_V2_T_PATH = '/home/ml_trainer/worm-project/pt-models/swin/swin_v2_t.pth'

# VIT_B_16_PATH = '/home/ml_trainer/worm-project/pt-models/vit/vit_b_16.pth'
# VIT_B_32_PATH = '/home/ml_trainer/worm-project/pt-models/vit/vit_b_32.pth'
# VIT_H_14_PATH = '/home/ml_trainer/worm-project/pt-models/vit/vit_h_14.pth'
# VIT_L_16_PATH = '/home/ml_trainer/worm-project/pt-models/vit/vit_l_16.pth'
# VIT_L_32_PATH = '/home/ml_trainer/worm-project/pt-models/vit/vit_l_32.pth'

# YOLO26N_CLS_PATH = '/home/ml_trainer/worm-project/pt-models/yolo/yolo26n-cls.pt'
# YOLO26S_CLS_PATH = '/home/ml_trainer/worm-project/pt-models/yolo/yolo26s-cls.pt'
# YOLO26M_CLS_PATH = '/home/ml_trainer/worm-project/pt-models/yolo/yolo26m-cls.pt'
# YOLO26L_CLS_PATH = '/home/ml_trainer/worm-project/pt-models/yolo/yolo26l-cls.pt'
# YOLO26X_CLS_PATH = '/home/ml_trainer/worm-project/pt-models/yolo/yolo26x-cls.pt'

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
# OUTPUT_DIR = '/home/ml_trainer/worm-project/worm-classification-model'
OUTPUT_DIR = '/scratch/lt200264-saiwat/worm-project/worm-classification-model'

# experiment name
# NOTE: CSV columns changed (precision-target metrics) -> use a NEW experiment name,
#       old CSVs / resume states from 'model-benchmark-v2' are not compatible
EXPERIMENT_NAME = 'model-benchmark-v3'

# dataset path
# CLS2_BLACKOUT = '/home/ml_trainer/worm-project/datasets/cls2_blackout/'
# CLS2_DIM = '/home/ml_trainer/worm-project/datasets/cls2_dim/'
# CLS2_RAW = '/home/ml_trainer/worm-project/datasets/cls2_raw/'
# CLS4_RAW = '/home/ml_trainer/worm-project/datasets/cls4_raw/'
# CLS5_RAW = '/home/ml_trainer/worm-project/datasets/cls5_raw/'
CLS2_BLACKOUT = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v3/views/cls2_blackout/'
CLS2_DIM = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v3/views/cls2_dim/'
CLS2_RAW = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v3/views/cls2_raw/'
CLS4_RAW = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v3/views/cls4_raw/'
CLS5_RAW = '/project/lt200264-saiwat/WormProject/data/release/worm-24022026-v3/views/cls5_raw/'

# name map
DATASETS = {
    '2class_blackout': CLS2_BLACKOUT,
    '2class_dim': CLS2_DIM,
    '2class_raw': CLS2_RAW,
    '4class_raw': CLS4_RAW,
    '5class_raw': CLS5_RAW,
}

# 2 class calculate
POSITIVE_CLASS_NAMES = ['Perfect']

# hardware & reproducibility
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = 4
# True  = cudnn deterministic algorithms (bit-exact reruns) but MUCH slower backward,
#         especially ConvNeXt 7x7 depthwise convs
# False = cudnn.benchmark picks the fastest algorithms (results still seeded, tiny run-to-run noise)
DETERMINISTIC = False
# TF32 matmul/conv for any op that stays fp32 under autocast (Ampere+: 3080 Ti, A100)
ALLOW_TF32 = True

# preprocessing & dataloader
IMG_SIZE = 224
UPSCALE = False
AUGMENT = True
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

# train
OPTIMIZER_NAME = 'AdamW'
FAMILY_LR = {}
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
START_FACTOR = 0.1
END_FACTOR = 1.0
ETA_MIN = 1e-5
USE_AMP = True
AMP_DTYPE = 'bf16'
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 10
# FSDP CPU offload (params / grads / AdamW step on CPU -> saves GPU memory, but slow on small-CPU machines)
#   'auto' = offload only when the model's params + grads + optimizer state would not fit
#            in FSDP_OFFLOAD_MEM_FRACTION of one GPU's memory
#   True   = always offload (old behaviour)
#   False  = never offload
FSDP_CPU_OFFLOAD = 'auto'
FSDP_OFFLOAD_MEM_FRACTION = 0.5

# model selection (Perfect = positive class)
# Threshold is searched EXACTLY on val (every unique P(Perfect)), frozen and reused for test / inference.
#   tgt   threshold = lowest threshold with Perfect precision >= TARGET_PRECISION (max recall, leak <= 5%)
#   f1opt threshold = best Perfect F1 (logged for reference only)
TARGET_PRECISION = 0.95
# what decides the best epoch (any 'val_*' key from evaluate.compute_metrics works), e.g.
#   'val_selection_score'   -> recall @ precision >= TARGET_PRECISION (recommended)
#                              (if the target is not reached: precision - 1, always < 0)
#   'val_ap'                -> PR-AUC of Perfect (threshold-free)
#   'val_f1opt_f1_perfect'  -> old behaviour (best Perfect F1)
# datasets without a Perfect class fall back to 'val_all_f1_macro'
SELECTION_METRIC = 'val_selection_score'

# inference benchmark
INFER_BATCH_SIZE = BATCH_SIZE
INFER_WARMUP = 20   # เอาไว้อุ่นเครื่อง hardware / library
INFER_ITERS = 100   # run จริง วัดจริง

# predictions
SAVE_VAL_PREDICTIONS = True
GALLERY_K = 50
GALLERY_COLS = 10