import os
import random

import numpy as np
import torch
import torch.distributed as dist


# torchrun จะ copy process ขึ้นมา พอแต่ละ process run ถึงตรงนี้แล้วจะรู้ว่า process จะไปอยู่ gpu ใบไหน
def init_distributed():
    # check ว่าใช้ torchrun หรือไม่
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        # แจกป้ายชื่อให้แต่ละร่าง
        rank = int(os.environ["RANK"])  # id ของ gpu แต่ละใบ: 0(main process), 1, 2, 3, ...
        world_size = int(os.environ["WORLD_SIZE"])  # จำนวน gpu
        local_rank = int(os.environ["LOCAL_RANK"])  # id ของ gpu ในเครื่องเดียวกัน(same node): 0, 1, 2, 3, ...

        # กระจาย process ไปยัง gpu แต่ละใบ
        cuda_index = local_rank % max(torch.cuda.device_count(), 1)  # set index ให้ process
        torch.cuda.set_device(cuda_index)
        device = torch.device(f"cuda:{cuda_index}")

        if not dist.is_initialized():
            # ทำให้ gpu แต่ละใบสามารถสื่อสารกันได้ (ครั้งแรกเท่านั้น)
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",  # ช่องทางแลกเปลี่ยนข้อมูลกัน gpu ใช้ nccl, cpu ใช้ gloo
                init_method="env://",  # จุดนัดพบ
                rank=rank,  # เลขประจำตัวของ process แต่ละร่างว่ามาอยู่ gpu ใบไหน
                world_size=world_size,  # จำนวน process ทั้งหมดที่รต้องรอให้ครบ
            )
        return True, rank, world_size, local_rank, device
    else:
        # กรณีรันแบบ python ปกติ (Single-GPU / CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device


# main process นอกจากจะคำนวณแล้ว จะ log output คนเดียว
def is_main_process(rank):
    return rank == 0


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # ล็อคการ Hash ของ Python
    # บังคับให้ CUDNN ใช้ Algorithm เดิมเสมอ
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id):  # dataloader ส่ง argument มา 1 ตัวเลยต้องมีอะไรมารองรับ
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_log_dir(base_dir, run_name):
    log_dir = os.path.join(base_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def make_run_id(tag, base_dirs):
    if isinstance(base_dirs, (str, os.PathLike)):
        base_dirs = [base_dirs]

    def taken(candidate):
        return any(os.path.isdir(os.path.join(bd, candidate)) for bd in base_dirs)

    if not taken(tag):
        return tag

    i = 1
    while taken(f"{tag}{i}"):
        i += 1
    return f"{tag}{i}"


def get_run_config_dict(
    model_name,
    seed,
    optimizer_name,
    scheduler_name,
    resolution_mode,
    img_scale,
    min_size,
    max_size,
    anchor_sizes,
    anchor_aspect_ratios,
    rpn_pre_nms_top_n_train,
    rpn_post_nms_top_n_train,
    rpn_nms_thresh,
    rpn_fg_iou_thresh,
    box_detections_per_img,
    box_score_thresh,
    box_nms_thresh,
    box_fg_iou_thresh,
    lr,
    warmup_epochs,
    epochs,
    batch_size,
    num_gpu,
    weight_decay,
    grad_accum_steps,
    use_amp,
):
    """Build the JSON-serialisable run-config snapshot. Every value is an explicit
    argument — the caller (WormMRCNN) is responsible for supplying them, this
    function never reaches into the config module itself."""
    return {
        "model": model_name,
        "environment": {
            "seed": seed,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
        },
        "resolution": {
            "mode": resolution_mode,
            "scale": img_scale,
            "min_size": min_size,
            "max_size": max_size,
        },
        "anchors": {
            "sizes": anchor_sizes,
            "aspect_ratios": anchor_aspect_ratios,
        },
        "rpn": {
            "pre_nms_train": rpn_pre_nms_top_n_train,
            "post_nms_train": rpn_post_nms_top_n_train,
            "nms_thresh": rpn_nms_thresh,
            "fg_iou_thresh": rpn_fg_iou_thresh,
        },
        "detection": {
            "max_detections": box_detections_per_img,
            "box_score_thresh": box_score_thresh,
            "box_nms_thresh": box_nms_thresh,
            "box_fg_iou_thresh": box_fg_iou_thresh,
        },
        "training": {
            "lr": lr,
            "warmup_epochs": warmup_epochs,
            "epochs": epochs,
            "batch_size_per_gpu": batch_size,
            "num_gpus": num_gpu,
            "effective_batch_size": batch_size * num_gpu * grad_accum_steps,
            "weight_decay": weight_decay,
            "grad_accum_steps": grad_accum_steps,
            "amp": use_amp,
        },
    }