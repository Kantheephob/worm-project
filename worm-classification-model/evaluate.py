import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix,
)

import config as c


def build_metrics(num_classes, device=c.DEVICE):
    metrics = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
        'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        'confusion_matrix': MulticlassConfusionMatrix(num_classes=num_classes),
    })
    return metrics.to(device)


def find_good_class_index(class_names, candidates=('good',)):
    normalized = {name.strip().lower(): i for i, name in enumerate(class_names)}
    for cand in candidates:
        idx = normalized.get(cand.strip().lower())
        if idx is not None:
            return idx
    return None


def build_label_collapse_map(class_names, good_idx):
    return torch.tensor([0 if i == good_idx else 1 for i in range(len(class_names))])