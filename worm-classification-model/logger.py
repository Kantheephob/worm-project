import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no display on HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from common import append_csv_row

_SCI_KEYS = {'current_lr', 'current_weight_decay', 'base_lr', 'min_lr', 'weight_decay', 'best_lr'}
_2DP_SUFFIXES = ('_sec', '_ms', '_fps', '_mb', '_m')


def _fmt(key, val):
    """Round numbers for the CSV; None -> empty cell."""
    if val is None:
        return ''
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        if key in _SCI_KEYS:
            return f'{val:.3e}'
        if key.endswith(_2DP_SUFFIXES) or 'threshold' in key:
            return round(val, 2)
        return round(val, 4)
    return val


def plot_confusion_matrix(cm, class_names, title):
    """Each cell: 'count\\n(row %)'. Colour = % of the true class (row)."""
    cm = np.asarray(cm.detach().cpu() if hasattr(cm, 'detach') else cm).astype(np.int64)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros(cm.shape, dtype=float), where=row_sum > 0) * 100

    n = cm.shape[0]
    size = max(6.0, 1.5 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(pct, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% of true class (row)')
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names, title=title,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    font = 11 if n <= 3 else (10 if n <= 5 else 8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cm[i, j]:,}\n({pct[i, j]:.1f}%)', ha='center', va='center', fontsize=font,
                    color='white' if pct[i, j] > 50 else 'black')

    fig.tight_layout()
    return fig


class TrainLogger:
    EPOCH_FIELDNAMES = [
        'epoch', 'current_lr', 'current_weight_decay',
        'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'val_prec', 'val_recall', 'val_f1',
        'val_f1_perfect',           # Perfect F1 (argmax)
        'val_best_threshold',       # best Perfect threshold this epoch
        'val_f1_perfect_at_thr',    # Perfect F1 using that threshold
        'epoch_time_sec',
    ]

    SUMMARY_FIELDNAMES = [
        'variant', 'dataset_name',
        # setting
        'img_size', 'effective_batch_size', 'num_gpus',
        'optimizer', 'lr_scheduler',
        'base_lr', 'min_lr', 'weight_decay',
        'warmup_epochs', 'warmup_ratio', 'warmup_start_factor', 'warmup_end_factor',
        'max_epochs', 'early_stop_patience', 'grad_clip_norm',
        'param_count_m', 'best_epoch', 'stopped_epoch', 'best_lr',
        'total_train_time_sec', 'seed', 'use_amp', 'amp_dtype', 'augment',
        'model_size_mb', 'inference_latency_ms', 'inference_fps',
        'optuna_tuned', 'optuna_trial',
        # results
        'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'val_prec', 'val_recall', 'val_f1',
        'test_loss', 'test_acc', 'test_prec', 'test_recall', 'test_f1',
        'val_acc_2class', 'val_prec_2class', 'val_recall_2class', 'val_f1_2class',
        'test_acc_2class', 'test_prec_2class', 'test_recall_2class', 'test_f1_2class',
        # threshold
        'perfect_threshold',
        'val_prec_perfect_at_thr', 'val_recall_perfect_at_thr', 'val_f1_perfect_at_thr',
        'test_prec_perfect_at_thr', 'test_recall_perfect_at_thr', 'test_f1_perfect_at_thr',
    ]

    _TB_EPOCH_TAGS = {
        'Loss/train': 'train_loss',
        'Loss/val': 'val_loss',
        'Accuracy/train': 'train_acc',
        'Accuracy/val': 'val_acc',
        'Precision/val': 'val_prec',
        'Recall/val': 'val_recall',
        'F1/val_macro': 'val_f1',
        'F1/val_perfect': 'val_f1_perfect',
        'F1/val_perfect_at_thr': 'val_f1_perfect_at_thr',
        'Threshold/val_best': 'val_best_threshold',
        'LR': 'current_lr',
        'WeightDecay': 'current_weight_decay',
        'Time/epoch_sec': 'epoch_time_sec',
    }

    def __init__(self, model_name, log_dir, tb_dir):
        self.model_name = model_name
        self.log_path = Path(log_dir) / f'{model_name}.csv'
        self.summary_path = Path(log_dir) / 'summary.csv'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        tb_run_dir = Path(tb_dir) / model_name
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_run_dir))

    def log_epoch(self, row):
        """row: dict with keys from EPOCH_FIELDNAMES (missing keys -> empty cell)."""
        epoch = row['epoch']
        csv_row = {k: _fmt(k, row.get(k)) for k in self.EPOCH_FIELDNAMES}
        append_csv_row(self.log_path, csv_row, self.EPOCH_FIELDNAMES)

        for tag, key in self._TB_EPOCH_TAGS.items():
            val = row.get(key)
            if val is not None:
                self.writer.add_scalar(tag, val, epoch)

    def log_summary(self, summary):
        """summary: dict, keys outside SUMMARY_FIELDNAMES are ignored."""
        csv_row = {k: _fmt(k, summary.get(k)) for k in self.SUMMARY_FIELDNAMES}
        append_csv_row(self.summary_path, csv_row, self.SUMMARY_FIELDNAMES)

    def log_confusion_matrix(self, conf_matrix, class_names, step, title='Confusion_Matrix', png_path=None):
        fig = plot_confusion_matrix(conf_matrix, class_names, title)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        if png_path is not None:
            Path(png_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=150)
        plt.close(fig)
        buf.seek(0)
        image_tensor = to_tensor(PIL.Image.open(buf).convert('RGB'))
        self.writer.add_image(title, image_tensor, step)

    def save_confusion_matrix_png(self, conf_matrix, class_names, title, png_path):
        fig = plot_confusion_matrix(conf_matrix, class_names, title)
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    def log_hparams(self, hparams, metrics):
        clean_hparams = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        self.writer.add_hparams(clean_hparams, clean_metrics, run_name='hparams')

    def close(self):
        self.writer.close()