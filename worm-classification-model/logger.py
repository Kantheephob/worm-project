import io
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from common import append_csv_row


class TrainLogger:
    EPOCH_FIELDNAMES = [
        'epoch', 'lr', 'train_loss', 'train_accuracy',
        'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
        'epoch_time_sec',
    ]
    
    SUMMARY_FIELDNAMES = [
        'variant', 'best_epoch', 'seed', 'total_train_time_sec', 'train_loss',
        'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
        'val_accuracy_2class', 'val_precision_2class', 'val_recall_2class', 'val_f1_2class',
        'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
        'test_accuracy_2class', 'test_precision_2class', 'test_recall_2class', 'test_f1_2class',
    ]

    def __init__(self, model_name, log_dir, tb_dir):
        self.model_name = model_name
        self.log_path = Path(log_dir) / f'{model_name}.csv'
        self.summary_path = Path(log_dir) / 'summary.csv'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        tb_run_dir = Path(tb_dir) / model_name
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_run_dir))

    def log_epoch(self, epoch, lr, train_loss, train_accuracy, val_loss, val_accuracy,
                  val_precision, val_recall, val_f1, epoch_time_sec):
        row = {
            'epoch': epoch,
            'lr': round(lr, 6),
            'train_loss': round(train_loss, 4),
            'train_accuracy': round(train_accuracy, 4),
            'val_loss': round(val_loss, 4),
            'val_accuracy': round(val_accuracy, 4),
            'val_precision': round(val_precision, 4),
            'val_recall': round(val_recall, 4),
            'val_f1': round(val_f1, 4),
            'epoch_time_sec': round(epoch_time_sec, 2),
        }
        append_csv_row(self.log_path, row, self.EPOCH_FIELDNAMES)

        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        self.writer.add_scalar('F1/val', val_f1, epoch)
        self.writer.add_scalar('LR', lr, epoch)

    def log_summary(self, best_epoch, seed, train_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1,
                 test_loss, test_accuracy, test_precision, test_recall, test_f1, total_train_time_sec,
                 test_accuracy_2class=None, test_precision_2class=None,
                 test_recall_2class=None, test_f1_2class=None,
                 val_accuracy_2class=None, val_precision_2class=None,
                 val_recall_2class=None, val_f1_2class=None):
        
        def safe_round(val, decimals=4):
            return round(val, decimals) if val is not None else ''

        summary = {
            'variant': self.model_name,
            'best_epoch': best_epoch,
            'seed': seed, 
            'total_train_time_sec': safe_round(total_train_time_sec, 2),
            'train_loss': safe_round(train_loss),
            'val_loss': safe_round(val_loss),
            'val_accuracy': safe_round(val_accuracy),
            'val_precision': safe_round(val_precision),
            'val_recall': safe_round(val_recall),
            'val_f1': safe_round(val_f1),
            'val_accuracy_2class': safe_round(val_accuracy_2class),
            'val_precision_2class': safe_round(val_precision_2class),
            'val_recall_2class': safe_round(val_recall_2class),
            'val_f1_2class': safe_round(val_f1_2class),
            'test_loss': safe_round(test_loss),
            'test_accuracy': safe_round(test_accuracy),
            'test_precision': safe_round(test_precision),
            'test_recall': safe_round(test_recall),
            'test_f1': safe_round(test_f1),
            'test_accuracy_2class': safe_round(test_accuracy_2class),
            'test_precision_2class': safe_round(test_precision_2class),
            'test_recall_2class': safe_round(test_recall_2class),
            'test_f1_2class': safe_round(test_f1_2class),
        }
        append_csv_row(self.summary_path, summary, self.SUMMARY_FIELDNAMES)

    def log_confusion_matrix(self, conf_matrix, class_names, step, title='Confusion_Matrix'):
        cm = conf_matrix.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names, title=title,
               ylabel='True label', xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image_tensor = to_tensor(PIL.Image.open(buf))
        self.writer.add_image(title, image_tensor, step)

    def log_hparams(self, hparams, metrics):
        clean_hparams = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        self.writer.add_hparams(clean_hparams, metrics, run_name='hparams')

    def close(self):
        self.writer.close()