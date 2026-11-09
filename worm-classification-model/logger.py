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
from evaluate import BIN_METRICS

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
        if 'threshold' in key:
            return round(val, 6)          # overconfident models need e.g. 0.999812
        if key.endswith(_2DP_SUFFIXES):
            return round(val, 2)
        return round(val, 4)
    return val


def split_metric_cols(split, ops):
    """Every metric column for one split ('val' / 'test'), most important first.
    ops = evaluate.target_ops(...)"""
    cols = [f'{split}_ap', f'{split}_roc_auc', f'{split}_loss',
            f'{split}_all_acc', f'{split}_all_prec_macro', f'{split}_all_recall_macro', f'{split}_all_f1_macro']
    # headline per operating point: threshold, target met?, Perfect precision / recall, leaked count
    for name, _, _ in ops:
        cols += [f'{split}_{name}_threshold', f'{split}_{name}_met', f'{split}_{name}_prec_perfect',
                 f'{split}_{name}_recall_perfect', f'{split}_{name}_fp']
    cols += [f'{split}_f1opt_threshold']
    # full blocks
    for name in ['argmax'] + [o[0] for o in ops] + ['f1opt']:
        for m in BIN_METRICS:
            col = f'{split}_{name}_{m}'
            if col not in cols:
                cols.append(col)
    return cols


def plot_confusion_matrix(cm, class_names, title, col_names=None):
    """Each cell: 'count\\n(row %)'. Colour = % of the true class (row).
    class_names = row labels (true). col_names = column labels (predicted), default = class_names.
    Non-square works too, e.g. by-class breakdown: rows = 5 true classes, cols = [Perfect, Others]."""
    cm = np.asarray(cm.detach().cpu() if hasattr(cm, 'detach') else cm).astype(np.int64)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros(cm.shape, dtype=float), where=row_sum > 0) * 100

    col_names = class_names if col_names is None else col_names
    n_rows, n_cols = cm.shape
    if n_rows == n_cols:                          # normal confusion matrix -> same size as before
        width = height = max(6.0, 1.5 * n_rows + 2)
    else:                                         # by-class breakdown (e.g. 5 x 2)
        width, height = max(6.0, 1.5 * n_cols + 3), max(4.5, 0.9 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(pct, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% of true class (row)')
    ax.set(xticks=np.arange(n_cols), yticks=np.arange(n_rows),
           xticklabels=col_names, yticklabels=class_names, title=title,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    n = max(n_rows, n_cols)
    font = 11 if n <= 3 else (10 if n <= 5 else 8)
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f'{cm[i, j]:,}\n({pct[i, j]:.1f}%)', ha='center', va='center', fontsize=font,
                    color='white' if pct[i, j] > 50 else 'black')

    fig.tight_layout()
    return fig


def plot_pr_curve(curve, points, title):
    """Precision-recall curve of the Perfect class with the operating points marked.
    curve  = {'rec': np.array, 'prec': np.array}
    points = [(label, recall, precision, kind), ...]  kind in {'argmax', 'precision', 'recall', 'f1opt'}"""
    style = {'argmax': ('s', '#555555'), 'precision': ('o', '#1f77b4'),
             'recall': ('^', '#d62728'), 'f1opt': ('D', '#2ca02c')}
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve['rec'], curve['prec'], color='#333333', lw=1.5, label='PR curve')
    # precision-locked labels above-right, recall-locked below-left (they sit on different parts
    # of the curve); argmax / best-F1 values go in the legend so they never cover other labels
    offset = {'precision': ((6, 6), 'left'), 'recall': ((-8, -24), 'right')}
    seen = set()
    for label, r, p, kind in points:
        marker, color = style.get(kind, ('o', 'black'))
        if kind in offset:
            (dx, dy), ha = offset[kind]
            ax.scatter([r], [p], marker=marker, s=60, color=color, zorder=3, edgecolor='white', linewidth=0.8)
            ax.annotate(f'{label}\nP={p:.2f} R={r:.2f}', (r, p), textcoords='offset points', xytext=(dx, dy),
                        fontsize=8, color=color, ha=ha)
            if kind not in seen:
                seen.add(kind)
                name = 'precision-locked (P ≥ target)' if kind == 'precision' else 'recall-locked (R ≥ target)'
                ax.scatter([], [], marker=marker, color=color, label=name)
        else:
            name = 'argmax (0.5)' if kind == 'argmax' else label
            ax.scatter([r], [p], marker=marker, s=60, color=color, zorder=3, edgecolor='white', linewidth=0.8,
                       label=f'{name}: P={p:.2f} R={r:.2f}')
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel='Recall',
           ylabel='Precision', title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    return fig


def _fig_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return to_tensor(PIL.Image.open(buf).convert('RGB'))


class TrainLogger:
    _TB_FIXED = {
        'LR': 'current_lr',
        'WeightDecay': 'current_weight_decay',
        'Time/epoch_sec': 'epoch_time_sec',
    }

    def __init__(self, model_name, log_dir, tb_dir, ops=()):
        self.model_name = model_name
        self.ops = list(ops)
        self.op_names = ['all', 'argmax', 'f1opt'] + [o[0] for o in self.ops]
        self.log_path = Path(log_dir) / f'{model_name}.csv'
        self.summary_path = Path(log_dir) / 'summary.csv'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.epoch_fieldnames = (
            ['epoch', 'current_lr', 'current_weight_decay', 'train_loss', 'train_acc']
            + split_metric_cols('val', self.ops)
            + ['epoch_time_sec']
        )
        self.summary_fieldnames = (
            ['model', 'variant', 'dataset_name',
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
             # selection
             'selection_metric', 'best_selection_value', 'precision_targets', 'recall_targets',
             # results
             'train_loss', 'train_acc']
            + split_metric_cols('val', self.ops)
            + split_metric_cols('test', self.ops)
        )

        tb_run_dir = Path(tb_dir) / model_name
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_run_dir))

    def _tb_tag(self, key):
        """'val_p95_prec_perfect' -> 'val_p95/prec_perfect'  |  'train_loss' -> 'train/loss'"""
        split, rest = key.split('_', 1)
        for op in self.op_names:
            if rest.startswith(op + '_'):
                return f'{split}_{op}/{rest[len(op) + 1:]}'
        return f'{split}/{rest}'

    def log_epoch(self, row):
        """row: dict with keys from epoch_fieldnames (missing keys -> empty cell)."""
        epoch = row['epoch']
        csv_row = {k: _fmt(k, row.get(k)) for k in self.epoch_fieldnames}
        append_csv_row(self.log_path, csv_row, self.epoch_fieldnames)

        for tag, key in self._TB_FIXED.items():
            val = row.get(key)
            if val is not None:
                self.writer.add_scalar(tag, val, epoch)
        for key, val in row.items():
            if key.startswith(('train_', 'val_')) and isinstance(val, (int, float)) and not isinstance(val, bool):
                self.writer.add_scalar(self._tb_tag(key), val, epoch)

    def log_summary(self, summary):
        """summary: dict, keys outside summary_fieldnames are ignored."""
        csv_row = {k: _fmt(k, summary.get(k)) for k in self.summary_fieldnames}
        append_csv_row(self.summary_path, csv_row, self.summary_fieldnames)

    def log_confusion_matrix(self, conf_matrix, class_names, step, title='Confusion_Matrix', png_path=None,
                             col_names=None):
        fig = plot_confusion_matrix(conf_matrix, class_names, title, col_names=col_names)
        image_tensor = _fig_to_tensor(fig)
        if png_path is not None:
            Path(png_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=150)
        plt.close(fig)
        self.writer.add_image(title, image_tensor, step)

    def save_confusion_matrix_png(self, conf_matrix, class_names, title, png_path, col_names=None):
        fig = plot_confusion_matrix(conf_matrix, class_names, title, col_names=col_names)
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    def save_pr_curve_png(self, curve, points, title, png_path, tb_tag=None, step=None):
        if curve is None:
            return
        fig = plot_pr_curve(curve, points, title)
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150)
        if tb_tag is not None:
            self.writer.add_image(tb_tag, _fig_to_tensor(fig), step if step is not None else 0)
        plt.close(fig)

    def log_hparams(self, hparams, metrics):
        clean_hparams = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        self.writer.add_hparams(clean_hparams, clean_metrics, run_name='hparams')

    def close(self):
        self.writer.close()