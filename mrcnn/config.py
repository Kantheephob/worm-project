import os
import torch
import sys
sys.path.insert(0, "/home/kklamcha/worm-project/")
import worm_project_config as wpf

# ---- data paths ----
LABELS_PATH = wpf.LABELS_PATH
ROI_PATH = wpf.ROI_PATH
MASKS_DIR = wpf.MASKS_DIR

# ---- class map ----
CLASS_NAMES = wpf.CLASS_NAMES
BINARY_CLASS_NAMES = wpf.BINARY_CLASS_NAMES
CLASS_COLORS = wpf.CLASS_COLORS

MRCNN_PT_PATH = "/project/lt200264-saiwat/pretrained_models/maskrcnn_resnet50_fpn_v2/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
CHECKPOINT_DIR = "/home/kklamcha/worm-project/mrcnn/ckpt"
TB_DIR = "/home/kklamcha/worm-project/mrcnn/tb"
ONNX_DIR = "/home/kklamcha/worm-project/mrcnn/onnx"

# ---- which label sets to train on, in order. Runs multi first, then binary. ----
DATASETS_TO_RUN = ["multi", "binary"]

# ---- device ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- dataset ----
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---- dataloader ----
NUM_GPU = int(os.environ.get("WORLD_SIZE", 1))
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4

# ---- color / inference defaults ---- (single source of truth = worm_project_config.py)
COLOR_ALPHA = wpf.COLOR_ALPHA
PREDICT_THRESHOLD = wpf.PREDICT_THRESHOLD
MASK_THRESHOLD = wpf.MASK_THRESHOLD

# ---- mixed precision (AMP) ----
USE_AMP = True

# ---- memory/experiment knobs ----
USE_BACKBONE_CHECKPOINTING = False
CHECKPOINT_SEGMENTS = 2

GRAD_ACCUM_STEPS = 4

# ---- experiment meta ----
# ตั้งชื่อ tag ให้ตรงกับรอบทดลองทุกครั้งก่อน submit job — ใช้เป็น run_id ตรงๆ (ไม่มี timestamp)
# ถ้าโฟลเดอร์ชื่อ tag นี้มีอยู่แล้วใน checkpoint/tensorboard/onnx dir จะ auto-increment ให้เอง
EXPERIMENT_TAG = "resize_by_2_detect_400_reduce_anchor_size_by_half"
MODEL_NAME = "mrcnn_rn50_fpn_v2"

# ---- optimization / schedule ----
EPOCHS = 100
WARMUP_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1

# ---- checkpointing / early stopping ----
SAVE_EVERY = 10
EARLY_STOP_PATIENCE = 10

# ---- reproducibility ----
SEED = 42

# ---- resume/finetune from an existing checkpoint instead of COCO weights ----
RESUME_FROM = None

# ---- run the held-out test split after training finishes ----
RUN_TEST_AFTER_TRAIN = True

# ---- Optimizer / Scheduler ----
OPTIMIZER_NAME = "AdamW"
SCHEDULER_NAME = "LinearWarmup_Cosine"

# ---- eval ----
MAX_DETECTION_THRESHOLDS = (1, 10, 500)

PRIMARY_METRIC = "segm_map_50"

# =====================================================================
# ---- input resolution ----
# =====================================================================
# "default" -> ใช้ resolution มาตรฐานของ torchvision (800, 1333) สำหรับรอบ baseline
# "custom"  -> ดึง shape จริงจากรูปตัวอย่าง (กล้องล็อกอยู่กับที่ ขนาดเท่ากันทุกใบ) คูณ IMG_SCALE
RESOLUTION_MODE = "custom"

# ใช้เฉพาะตอน RESOLUTION_MODE == "custom" — path รูป raw_img_path จริง 1 ใบพอ
SAMPLE_IMG_PATH = '/project/lt200264-saiwat/WormProject/data/worm-24022026/raw/worm-000001.jpg'
IMG_SCALE = 1 / 2


def _infer_min_max_size(sample_path, scale):
    from PIL import Image
    with Image.open(sample_path) as img:
        w, h = img.size
    short_side, long_side = min(h, w), max(h, w)
    return int(short_side * scale), int(long_side * scale)


if RESOLUTION_MODE == "default":
    MIN_SIZE = 800
    MAX_SIZE = 1333
elif RESOLUTION_MODE == "custom":
    MIN_SIZE, MAX_SIZE = _infer_min_max_size(SAMPLE_IMG_PATH, IMG_SCALE)
else:
    raise ValueError(f"Unknown RESOLUTION_MODE: {RESOLUTION_MODE!r}")

# =====================================================================
# ---- anchor generator ----
# =====================================================================
ANCHOR_SIZES = (16, 32, 64, 128, 256)
ANCHOR_ASPECT_RATIOS = (0.5, 1.0, 2.0)

# ---- RPN Parameters ----
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 2000
RPN_POST_NMS_TOP_N_TEST = 1000
RPN_NMS_THRESH = 0.7
RPN_FG_IOU_THRESH = 0.7
RPN_BG_IOU_THRESH = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION = 0.5
RPN_SCORE_THRESH = 0.0

# ---- Box Parameters ----
BOX_SCORE_THRESH = 0.05
BOX_NMS_THRESH = 0.5
BOX_FG_IOU_THRESH = 0.5
BOX_BG_IOU_THRESH = 0.5
BOX_BATCH_SIZE_PER_IMAGE = 512
BOX_POSITIVE_FRACTION = 0.25
BOX_DETECTIONS_PER_IMG = 400

# ---- onnx setting ----
ONNX_OPSET_VER = 17