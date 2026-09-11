from torchmetrics.detection import MeanAveragePrecision



def build_metric(max_detection_thresholds, class_metrics=True, sync_on_compute=True):
    return MeanAveragePrecision(
        iou_type=["bbox", "segm"],
        class_metrics=class_metrics,
        max_detection_thresholds=list(max_detection_thresholds),
        sync_on_compute=sync_on_compute,
    )


def extract_summary(computed, mar_key):
    """Overall (not per-class) scalar metrics for both box and mask."""
    summary = {}
    for iou_type in ("bbox", "segm"):
        summary[f"{iou_type}_map"] = computed[f"{iou_type}_map"].item()
        summary[f"{iou_type}_map_50"] = computed[f"{iou_type}_map_50"].item()
        summary[f"{iou_type}_map_75"] = computed[f"{iou_type}_map_75"].item()
        summary[f"{iou_type}_{mar_key}"] = computed[f"{iou_type}_{mar_key}"].item()
    return summary


def extract_per_class(computed, class_map, mar_key, class_names=None):
    """Per-class AP and AR@mar_key for both box and mask.

    class_map maps {original_class_value: model_label_idx}; torchmetrics reports
    per-class rows against a shared 'classes' tensor of model_label_idx values,
    so we invert class_map to recover each row's original_class_value, then look
    that up in class_names (e.g. config.CLASS_NAMES) for a human-readable label.
    Falls back to the raw original_class_value (as a string) if class_names is
    None or doesn't have an entry for it.
    """
    idx_to_orig = {v: k for k, v in class_map.items()}
    classes = computed.get("classes")
    if classes is None:
        return {}

    per_class = {}
    for iou_type in ("bbox", "segm"):
        map_per_class = computed.get(f"{iou_type}_map_per_class")
        mar_per_class = computed.get(f"{iou_type}_{mar_key}_per_class")
        if map_per_class is None:
            continue
        for i, cls_idx in enumerate(classes.tolist()):
            orig_value = idx_to_orig.get(cls_idx, cls_idx)
            if class_names is not None and orig_value in class_names:
                name = class_names[orig_value]
            else:
                name = str(orig_value)
            per_class[f"{iou_type}/{name}_map"] = map_per_class[i].item()
            if mar_per_class is not None:
                per_class[f"{iou_type}/{name}_{mar_key}"] = mar_per_class[i].item()
    return per_class


def log_metrics_to_console(prefix, summary):
    parts = " | ".join(f"{k}={v:.4f}" for k, v in summary.items())
    print(f"  [{prefix}] {parts}", flush=True)


def log_metrics_to_tensorboard(writer, summary, per_class, step, split):
    """split: 'val' or 'test' — used as the scalar name prefix."""
    for k, v in summary.items():
        writer.add_scalar(f"{split}/{k}", v, step)
    for k, v in per_class.items():
        writer.add_scalar(f"{split}_per_class/{k}", v, step)