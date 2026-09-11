import json
import os
import time

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt

import config as c
from common import (
    init_distributed,
    is_main_process,
    log,
    make_log_dir,
    make_run_id,
    set_seed,
    get_run_config_dict,
)
from dataset import build_dataset_loaders, get_transforms
from model import get_mrcnn_model, enable_backbone_checkpointing, unwrap_model
from postprocess import binarize_mask_preds, to_cpu_dicts
from metrics import (
    build_metric,
    extract_summary,
    extract_per_class,
    log_metrics_to_console,
    log_metrics_to_tensorboard,
)
from export_onnx import export_onnx as _export_onnx_fn


class WormMRCNN:
    """Object wrapper around the worm Mask R-CNN pipeline.

    Usage:
        model = WormMRCNN(dataset_choice="multi")
        model.train()
        model.test()
        model.export_onnx()

        # later, for inference only:
        detector = WormMRCNN(dataset_choice="multi")
        detector.load("checkpoint/.../best.pt")
        img, results = detector.predict("some_image.jpg")
        detector.show_result("some_image.jpg", save_path="out.png")
    """

    def __init__(
        self,
        dataset_choice="multi",
        # ---- paths ----
        label_path=c.LABELS_PATH,
        roi_path=c.ROI_PATH,
        mask_dir=c.MASKS_DIR,
        coco_pretrained_path=c.MRCNN_PT_PATH,
        checkpoint_dir=c.CHECKPOINT_DIR,
        tb_dir=c.TB_DIR,
        onnx_dir=c.ONNX_DIR,
        # ---- dataset / loader ----
        train_split=c.TRAIN_SPLIT,
        val_split=c.VAL_SPLIT,
        batch_size=c.BATCH_SIZE,
        val_batch_size=c.VAL_BATCH_SIZE,
        num_workers=c.NUM_WORKERS,
        # ---- input resolution ----
        resolution_mode=c.RESOLUTION_MODE,
        img_scale=c.IMG_SCALE,
        min_size=c.MIN_SIZE,
        max_size=c.MAX_SIZE,
        # ---- anchors ----
        anchor_sizes=c.ANCHOR_SIZES,
        anchor_aspect_ratios=c.ANCHOR_ASPECT_RATIOS,
        # ---- rpn ----
        rpn_pre_nms_top_n_train=c.RPN_PRE_NMS_TOP_N_TRAIN,
        rpn_pre_nms_top_n_test=c.RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_train=c.RPN_POST_NMS_TOP_N_TRAIN,
        rpn_post_nms_top_n_test=c.RPN_POST_NMS_TOP_N_TEST,
        rpn_nms_thresh=c.RPN_NMS_THRESH,
        rpn_fg_iou_thresh=c.RPN_FG_IOU_THRESH,
        rpn_bg_iou_thresh=c.RPN_BG_IOU_THRESH,
        rpn_batch_size_per_image=c.RPN_BATCH_SIZE_PER_IMAGE,
        rpn_positive_fraction=c.RPN_POSITIVE_FRACTION,
        rpn_score_thresh=c.RPN_SCORE_THRESH,
        # ---- box ----
        box_score_thresh=c.BOX_SCORE_THRESH,
        box_nms_thresh=c.BOX_NMS_THRESH,
        box_detections_per_img=c.BOX_DETECTIONS_PER_IMG,
        box_fg_iou_thresh=c.BOX_FG_IOU_THRESH,
        box_bg_iou_thresh=c.BOX_BG_IOU_THRESH,
        box_batch_size_per_image=c.BOX_BATCH_SIZE_PER_IMAGE,
        box_positive_fraction=c.BOX_POSITIVE_FRACTION,
        # ---- optimization / schedule ----
        epochs=c.EPOCHS,
        warmup_epochs=c.WARMUP_EPOCHS,
        lr=c.LR,
        weight_decay=c.WEIGHT_DECAY,
        grad_clip_norm=c.GRAD_CLIP_NORM,
        grad_accum_steps=c.GRAD_ACCUM_STEPS,
        use_amp=c.USE_AMP,
        save_every=c.SAVE_EVERY,
        early_stop_patience=c.EARLY_STOP_PATIENCE,
        mask_threshold=c.MASK_THRESHOLD,
        # ---- eval ----
        max_detection_thresholds=c.MAX_DETECTION_THRESHOLDS,
        primary_metric=c.PRIMARY_METRIC,
        # ---- memory knobs ----
        use_backbone_checkpointing=c.USE_BACKBONE_CHECKPOINTING,
        checkpoint_segments=c.CHECKPOINT_SEGMENTS,
        # ---- experiment meta ----
        model_name=c.MODEL_NAME,
        experiment_tag=c.EXPERIMENT_TAG,
        optimizer_name=c.OPTIMIZER_NAME,
        scheduler_name=c.SCHEDULER_NAME,
        seed=c.SEED,
        run_id=None,
        device=None,
        class_names=None,
    ):
        self.dataset_choice = dataset_choice
        if class_names is not None:
            self.class_names = class_names
        elif dataset_choice == "binary":
            self.class_names = c.BINARY_CLASS_NAMES
        else:
            self.class_names = c.CLASS_NAMES

        # paths
        self.label_path = label_path
        self.roi_path = roi_path
        self.mask_dir = mask_dir
        self.coco_pretrained_path = coco_pretrained_path
        self.checkpoint_dir = checkpoint_dir
        self.tb_dir = tb_dir
        self.onnx_dir = onnx_dir

        # dataset / loader
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        # resolution
        self.resolution_mode = resolution_mode
        self.img_scale = img_scale
        self.min_size = min_size
        self.max_size = max_size

        # anchors
        self.anchor_sizes = anchor_sizes
        self.anchor_aspect_ratios = anchor_aspect_ratios

        # rpn
        self.rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.rpn_post_nms_top_n_train = rpn_post_nms_top_n_train
        self.rpn_post_nms_top_n_test = rpn_post_nms_top_n_test
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_fg_iou_thresh = rpn_fg_iou_thresh
        self.rpn_bg_iou_thresh = rpn_bg_iou_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction
        self.rpn_score_thresh = rpn_score_thresh

        # box
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img
        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction

        # optimization / schedule
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.use_amp = use_amp
        self.save_every = save_every
        self.early_stop_patience = early_stop_patience
        self.mask_threshold = mask_threshold

        # eval
        self.max_detection_thresholds = tuple(max_detection_thresholds)
        self.mar_key = f"mar_{self.max_detection_thresholds[-1]}"
        self.primary_metric = primary_metric

        # memory knobs
        self.use_backbone_checkpointing = use_backbone_checkpointing
        self.checkpoint_segments = checkpoint_segments

        # experiment meta
        self.model_name = model_name
        self.experiment_tag = experiment_tag
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.seed = seed

        # distributed setup
        self.is_distributed, self.rank, self.world_size, self.local_rank, auto_device = init_distributed()
        self.device = device if device is not None else auto_device
        self.cuda_index = self.device.index if self.device.type == "cuda" else 0
        self.num_gpu = self.world_size

        # run_id = experiment_tag itself, no timestamp. If a directory with that name
        # already exists under checkpoint/tensorboard/onnx dirs, auto-increments
        # (baseline -> baseline1 -> baseline2 ...) so forgetting to change
        # EXPERIMENT_TAG before a re-run never silently overwrites a previous run.
        self.run_id = run_id or make_run_id(
            tag=self.experiment_tag,
            base_dirs=[self.checkpoint_dir, self.tb_dir, self.onnx_dir],
        )

        # state populated by train() / load()
        self.model = None
        self.class_map = None
        self.idx_to_class = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.run_checkpoint_dir = None
        self.tb_log_dir = None
        self.best_ckpt_path = None
        self.best_metric_value = 0.0

        self.transforms = get_transforms(train=False)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def log_info(self, msg):
        log(msg, self.rank)

    def _build_loaders(self):
        self.train_loader, self.val_loader, self.test_loader, self.train_sampler = build_dataset_loaders(
            dataset_choice=self.dataset_choice,
            label_path=self.label_path,
            roi_path=self.roi_path,
            mask_dir=self.mask_dir,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            train_split=self.train_split,
            val_split=self.val_split,
            seed=self.seed,
            is_distributed=self.is_distributed,
            world_size=self.world_size,
            rank=self.rank,
        )
        self.class_map = self.train_loader.dataset.class_map
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        self.idx_to_class[0] = "Background"
        return self.train_loader, self.val_loader, self.test_loader, self.train_sampler

    def _build_model(self, num_cls, pt_path=None, device=None):
        device = device or self.device
        model = get_mrcnn_model(
            num_cls=num_cls,
            device=device,
            min_size=self.min_size,
            max_size=self.max_size,
            anchor_sizes=self.anchor_sizes,
            anchor_aspect_ratios=self.anchor_aspect_ratios,
            rpn_pre_nms_top_n_train=self.rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=self.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=self.rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=self.rpn_post_nms_top_n_test,
            rpn_nms_thresh=self.rpn_nms_thresh,
            rpn_fg_iou_thresh=self.rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=self.rpn_bg_iou_thresh,
            rpn_batch_size_per_image=self.rpn_batch_size_per_image,
            rpn_positive_fraction=self.rpn_positive_fraction,
            rpn_score_thresh=self.rpn_score_thresh,
            box_score_thresh=self.box_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            box_detections_per_img=self.box_detections_per_img,
            box_fg_iou_thresh=self.box_fg_iou_thresh,
            box_bg_iou_thresh=self.box_bg_iou_thresh,
            box_batch_size_per_image=self.box_batch_size_per_image,
            box_positive_fraction=self.box_positive_fraction,
            coco_pretrained_path=self.coco_pretrained_path,
            pt_path=pt_path,
        )
        if self.use_backbone_checkpointing:
            model = enable_backbone_checkpointing(model, segments=self.checkpoint_segments)
        return model

    def _run_config_dict(self):
        return get_run_config_dict(
            model_name=self.model_name,
            seed=self.seed,
            optimizer_name=self.optimizer_name,
            scheduler_name=self.scheduler_name,
            resolution_mode=self.resolution_mode,
            img_scale=self.img_scale,
            min_size=self.min_size,
            max_size=self.max_size,
            anchor_sizes=self.anchor_sizes,
            anchor_aspect_ratios=self.anchor_aspect_ratios,
            rpn_pre_nms_top_n_train=self.rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=self.rpn_post_nms_top_n_train,
            rpn_nms_thresh=self.rpn_nms_thresh,
            rpn_fg_iou_thresh=self.rpn_fg_iou_thresh,
            box_detections_per_img=self.box_detections_per_img,
            box_score_thresh=self.box_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            box_fg_iou_thresh=self.box_fg_iou_thresh,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_gpu=self.num_gpu,
            weight_decay=self.weight_decay,
            grad_accum_steps=self.grad_accum_steps,
            use_amp=self.use_amp,
        )

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def train(self, resume_from=None):
        set_seed(self.seed)

        if self.train_loader is None:
            self._build_loaders()

        num_cls = len(self.class_map) + 1

        self.run_checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_id, self.dataset_choice)
        if is_main_process(self.rank):
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)
            config_save_path = os.path.join(self.run_checkpoint_dir, "config.json")
            with open(config_save_path, "w") as f:
                json.dump(self._run_config_dict(), f, indent=4)
            self.log_info(f"-> Created run directory: {self.run_checkpoint_dir}")
            self.log_info(f"-> Saved run config to: {config_save_path}")

        writer = None
        if is_main_process(self.rank):
            self.tb_log_dir = make_log_dir(base_dir=os.path.join(self.tb_dir, self.run_id), run_name=self.dataset_choice)
            writer = SummaryWriter(log_dir=self.tb_log_dir)

        model = self._build_model(num_cls, pt_path=resume_from)
        model.to(self.device)

        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.cuda_index], output_device=self.cuda_index, find_unused_parameters=False)

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.epochs - self.warmup_epochs), eta_min=1e-6)
        lr_scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_epochs]
        )

        metric = build_metric(self.max_detection_thresholds, class_metrics=True)
        scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.best_metric_value = 0.0
        epochs_no_improve = 0
        self.best_ckpt_path = None

        for epoch in range(self.epochs):
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            model.train()
            training_loss, n = 0.0, 0
            start_train_time = time.time()
            epoch_loss_dict = {}

            optimizer.zero_grad(set_to_none=True)
            num_batches = len(self.train_loader)

            for step, (imgs, targets) in enumerate(self.train_loader):
                imgs = [img.to(self.device) for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                is_last_micro_step = ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == num_batches)
                sync_context = (
                    model.no_sync()
                    if (self.is_distributed and not is_last_micro_step and hasattr(model, "no_sync"))
                    else torch.enable_grad()
                )

                with sync_context:
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        loss_dict = model(imgs, targets)
                        mean_loss = sum(loss for loss in loss_dict.values())
                        loss_to_backprop = mean_loss / self.grad_accum_steps

                    scaler.scale(loss_to_backprop).backward()

                batch_size = len(imgs)
                # Track the un-scaled loss for logging purposes.
                training_loss += mean_loss.item() * batch_size
                n += batch_size

                for k, v in loss_dict.items():
                    if k not in epoch_loss_dict:
                        epoch_loss_dict[k] = 0.0
                    epoch_loss_dict[k] += v.item() * batch_size

                if is_last_micro_step:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(parameters, max_norm=self.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step()
            epoch_train_time = time.time() - start_train_time

            if self.is_distributed:
                stats = torch.tensor([training_loss, float(n)], device=self.device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)  # รวมค่า loss กลับจากแต่ละ gpu
                training_loss, n = stats[0].item(), stats[1].item()

                for k in epoch_loss_dict.keys():
                    loss_tensor = torch.tensor(epoch_loss_dict[k], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    epoch_loss_dict[k] = loss_tensor.item()

            train_avg_loss = training_loss / n
            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process(self.rank):
                writer.add_scalar("Loss/train_total", train_avg_loss, epoch)
                writer.add_scalar("LR/current", current_lr, epoch)
                writer.add_scalar("Time/train_epoch_seconds", epoch_train_time, epoch)

                for k, v in epoch_loss_dict.items():
                    writer.add_scalar(f"Loss/{k}", v / n, epoch)

            del loss_dict, mean_loss
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            model.eval()
            metric.reset()
            start_val_time = time.time()

            with torch.no_grad():
                for imgs, targets in self.val_loader:
                    imgs = [img.to(self.device) for img in imgs]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        preds = unwrap_model(model)(imgs)
                    preds_cpu = to_cpu_dicts(binarize_mask_preds(preds, threshold=self.mask_threshold))
                    targets_cpu = to_cpu_dicts(targets)
                    metric.update(preds_cpu, targets_cpu)
                    del preds, preds_cpu, targets_cpu

            epoch_val_time = time.time() - start_val_time
            val_computed = metric.compute()
            val_summary = extract_summary(val_computed, self.mar_key)
            val_per_class = extract_per_class(val_computed, self.class_map, self.mar_key, class_names=self.class_names)
            val_metric_value = val_summary[self.primary_metric]

            if is_main_process(self.rank):
                writer.add_scalar("Time/val_epoch_seconds", epoch_val_time, epoch)
                log_metrics_to_tensorboard(writer, val_summary, val_per_class, epoch, split="val")
                print(
                    f"{'=' * 50} Epoch {epoch + 1}/{self.epochs} {'=' * 50}\n"
                    f"Train: loss={train_avg_loss:.4f} | time={epoch_train_time:.2f}s | lr={current_lr:.2e}\n"
                    f"Val:   segm_mAP={val_summary['segm_map']:.4f} | segm_mAP50={val_summary['segm_map_50']:.4f} | "
                    f"segm_{self.mar_key}={val_summary[f'segm_{self.mar_key}']:.4f} | "
                    f"bbox_mAP={val_summary['bbox_map']:.4f} | bbox_{self.mar_key}={val_summary[f'bbox_{self.mar_key}']:.4f} | "
                    f"time={epoch_val_time:.2f}s",
                    flush=True,
                )
                if val_per_class:
                    log_metrics_to_console(f"val per-class (epoch {epoch + 1})", val_per_class)

            if is_main_process(self.rank) and (epoch + 1) % self.save_every == 0:
                save_path = os.path.join(self.run_checkpoint_dir, f"epoch{epoch + 1}.pt")
                torch.save(
                    {
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "epoch": epoch + 1,
                        "primary_metric": self.primary_metric,
                        "primary_metric_value": val_metric_value,
                        "val_summary": val_summary,
                        "class_map": self.class_map,
                    },
                    save_path,
                )
                print(f"  -> saved checkpoint: {save_path}\n", flush=True)

            if val_metric_value > self.best_metric_value:
                self.best_metric_value = val_metric_value
                epochs_no_improve = 0
                self.best_ckpt_path = os.path.join(self.run_checkpoint_dir, "best.pt")
                if is_main_process(self.rank):
                    torch.save(
                        {
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "epoch": epoch + 1,
                            "primary_metric": self.primary_metric,
                            "primary_metric_value": val_metric_value,
                            "val_summary": val_summary,
                            "class_map": self.class_map,
                        },
                        self.best_ckpt_path,
                    )
                    print(
                        f"  -> new best model ({self.primary_metric}={val_metric_value:.4f}), "
                        f"saved to {self.best_ckpt_path}\n",
                        flush=True,
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stop_patience:
                    if is_main_process(self.rank):
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(no {self.primary_metric} improvement for {self.early_stop_patience} epochs)\n",
                            flush=True,
                        )
                    break

            torch.cuda.empty_cache()

        if is_main_process(self.rank) and writer is not None:
            hparam_dict = {
                "model": self.model_name,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "train_batch_size": self.batch_size * self.num_gpu,
                "val_batch_size": self.val_batch_size,
                "test_batch_size": self.val_batch_size,
                "epochs": self.epochs,
                "img_scale": self.img_scale,
                "anchor_sizes": str(self.anchor_sizes),
                "anchor_ratios": str(self.anchor_aspect_ratios),
                "amp": self.use_amp,
            }
            metric_dict = {f"hparam/best_{self.primary_metric}": self.best_metric_value}
            writer.add_hparams(hparam_dict, metric_dict, run_name=".")
            writer.close()

        self.model = unwrap_model(model)
        return self.model, self.best_ckpt_path, self.run_checkpoint_dir, self.tb_log_dir

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------
    def test(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.best_ckpt_path
        if checkpoint_path is None:
            raise ValueError("No checkpoint_path given and no best checkpoint recorded — call .train() first or pass checkpoint_path explicitly.")

        if self.test_loader is None:
            self._build_loaders()

        print(
            f"{'=' * 20} Testing dataset: {self.dataset_choice} {'=' * 20}\n"
            f"  best checkpoint: {checkpoint_path}",
            flush=True,
        )

        num_cls = len(self.test_loader.dataset.class_map) + 1
        model = self._build_model(num_cls, pt_path=checkpoint_path)
        model.to(self.device)
        model.eval()

        metric = build_metric(self.max_detection_thresholds, class_metrics=True, sync_on_compute=False)

        with torch.no_grad():
            for imgs, targets in self.test_loader:
                imgs = [img.to(self.device) for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds = model(imgs)
                preds_cpu = to_cpu_dicts(binarize_mask_preds(preds, threshold=self.mask_threshold))
                targets_cpu = to_cpu_dicts(targets)
                metric.update(preds_cpu, targets_cpu)
                del preds, preds_cpu, targets_cpu

        test_computed = metric.compute()
        test_summary = extract_summary(test_computed, self.mar_key)
        test_per_class = extract_per_class(test_computed, self.class_map, self.mar_key, class_names=self.class_names)

        print(
            f"  [{self.dataset_choice}] TEST  segm_mAP={test_summary['segm_map']:.4f} | "
            f"segm_mAP50={test_summary['segm_map_50']:.4f} | segm_{self.mar_key}={test_summary[f'segm_{self.mar_key}']:.4f} | "
            f"bbox_mAP={test_summary['bbox_map']:.4f} | bbox_{self.mar_key}={test_summary[f'bbox_{self.mar_key}']:.4f}\n",
            flush=True,
        )
        if test_per_class:
            log_metrics_to_console(f"{self.dataset_choice} test per-class", test_per_class)

        if self.tb_log_dir is not None:
            writer = SummaryWriter(log_dir=self.tb_log_dir)
            log_metrics_to_tensorboard(writer, test_summary, test_per_class, 0, split="test")
            writer.close()

        save_dir = self.run_checkpoint_dir or self.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt_data = torch.load(checkpoint_path, map_location=self.device)
        final_save_path = os.path.join(save_dir, "best_tested.pt")
        torch.save(
            {
                "model_state_dict": best_ckpt_data.get("model_state_dict", best_ckpt_data),
                "epoch": best_ckpt_data.get("epoch"),
                "primary_metric": best_ckpt_data.get("primary_metric", self.primary_metric),
                "primary_metric_value": best_ckpt_data.get("primary_metric_value"),
                "val_summary": best_ckpt_data.get("val_summary"),
                "class_map": best_ckpt_data.get("class_map"),
                "test_summary": test_summary,
                "test_per_class": test_per_class,
                "source_checkpoint": checkpoint_path,
            },
            final_save_path,
        )

        summary_path = os.path.join(save_dir, "test_results.txt")
        with open(summary_path, "a") as f:
            f.write(
                f"dataset={self.dataset_choice} | source_checkpoint={checkpoint_path} | "
                f"final_checkpoint={final_save_path} | "
                f"test_segm_mAP={test_summary['segm_map']:.4f} | test_segm_mAP50={test_summary['segm_map_50']:.4f} | "
                f"test_segm_{self.mar_key}={test_summary[f'segm_{self.mar_key}']:.4f} | "
                f"test_bbox_mAP={test_summary['bbox_map']:.4f} | test_bbox_{self.mar_key}={test_summary[f'bbox_{self.mar_key}']:.4f}\n"
            )

        print(f"  -> saved final tested model to {final_save_path}", flush=True)
        print(f"  -> appended summary to {summary_path}\n", flush=True)

        self.model = unwrap_model(model)
        return {"summary": test_summary, "per_class": test_per_class}

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_onnx(self, checkpoint_path=None, onnx_path=None, opset_version=13, export_device=torch.device("cpu")):
        checkpoint_path = checkpoint_path or self.best_ckpt_path
        if checkpoint_path is None:
            raise ValueError("No checkpoint_path given and no best checkpoint recorded — call .train() first or pass checkpoint_path explicitly.")

        if onnx_path is None:
            onnx_save_dir = os.path.join(self.onnx_dir, self.run_id, self.dataset_choice)
            os.makedirs(onnx_save_dir, exist_ok=True)
            onnx_path = os.path.join(onnx_save_dir, "best.onnx")

        def build_model_fn(num_cls, pt_path, device):
            return self._build_model(num_cls, pt_path=pt_path, device=device)

        _export_onnx_fn(
            pt_path=checkpoint_path,
            onnx_path=onnx_path,
            min_size=self.min_size,
            max_size=self.max_size,
            build_model_fn=build_model_fn,
            device=export_device,
            opset_version=opset_version,
        )
        return onnx_path

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def load(self, checkpoint_path, device=None):
        """Load trained weights for inference (predict/visualize/show_result)."""
        device = device or self.device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        class_map = checkpoint["class_map"]
        num_cls = len(class_map) + 1

        model = self._build_model(num_cls, pt_path=None, device=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.class_map = class_map
        self.idx_to_class = {v: k for k, v in class_map.items()}
        self.idx_to_class[0] = "Background"
        self.device = device
        return self.model

    def predict(self, img_path, threshold=c.PREDICT_THRESHOLD):
        if self.model is None:
            raise ValueError("No model loaded — call .load(checkpoint_path) or .train() first.")

        tensor_img = read_image(img_path)
        tensor_img = self.transforms(tensor_img).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model([tensor_img])[0]

        results = {k: v.cpu() for k, v in pred.items()}
        keep_idx = results["scores"] > threshold

        filtered_results = {
            "boxes": results["boxes"][keep_idx],
            "labels": results["labels"][keep_idx],
            "scores": results["scores"][keep_idx],
            "masks": results["masks"][keep_idx],
        }
        filtered_results["class_names"] = [
            self.idx_to_class.get(l.item()) for l in filtered_results["labels"]
        ]

        return tensor_img.cpu(), filtered_results

    def visualize(self, img_tensor, results, mask_threshold=c.MASK_THRESHOLD, mask_alpha=c.COLOR_ALPHA, save_path=None):
        drawn_img = img_tensor.clone()
        bboxes = results["boxes"]
        masks = results["masks"]
        scores = results["scores"]
        class_names = results["class_names"]

        if len(bboxes) > 0:
            # set color
            if self.dataset_choice == "binary":
                name_to_color = {name: c.CLASS_COLORS[idx].lower() for idx, name in c.BINARY_CLASS_NAMES.items()}
            else:
                name_to_color = {name: c.CLASS_COLORS[idx].lower() for idx, name in c.CLASS_NAMES.items()}

            instance_colors = [name_to_color.get(name, "white") for name in class_names]

            # draw mask
            bool_masks = masks.squeeze(1) > mask_threshold  # squeeze(1) ตัดมิติที่มีขนาด = 1 ออก (Channel) -> แปลงเป็น boolean mask
            drawn_img = draw_segmentation_masks(
                drawn_img,
                bool_masks,
                alpha=mask_alpha,
                colors=instance_colors, 
            )

            # draw bbox
            labels_str = [f"{class_names[i]}: {scores[i]:.2f}" for i in range(len(bboxes))]
            drawn_img = draw_bounding_boxes(
                drawn_img,
                bboxes,
                labels=labels_str,
                colors=instance_colors, 
                width=2,
            )

        np_img = drawn_img.permute(1, 2, 0).numpy()  # C H W -> H W C

        plt.figure(figsize=(20, 20))
        plt.imshow(np_img)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Saved to {save_path}")
        plt.show()
        plt.close()

    def show_result(self, img_path, threshold=c.PREDICT_THRESHOLD, save_path=None):
        img, results = self.predict(img_path, threshold=threshold)
        self.visualize(img, results, save_path=save_path)