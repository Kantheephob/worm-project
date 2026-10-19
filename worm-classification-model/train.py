import json
import argparse
from pathlib import Path

import torch

from classifier import WormClassificationModel
from common import atomic_json_dump
import config as c


def main():
    parser = argparse.ArgumentParser(description='Train one dataset x one model (or family) on LANTA / lab PC')
    parser.add_argument('--dataset', '-ds', type=str, required=True, choices=list(c.DATASETS.keys()),
                        help=f'Dataset key. Options: {list(c.DATASETS.keys())}')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model name or family (e.g. resnet18, resnet, vit, swin...). '
                             'A family name trains all its variants in this one job.')
    parser.add_argument('--experiment', '-e', type=str, default=c.EXPERIMENT_NAME,
                        help='Experiment name. Rerunning with the same name resumes instead of '
                             'creating a new folder.')
    parser.add_argument('--num-gpus', type=int, default=c.NUM_GPUS, help='GPUs to use (default: config.NUM_GPUS)')
    # optional overrides (handy for a quick smoke test)
    parser.add_argument('--epochs', type=int, default=None, help='Override EPOCHS (e.g. 2 for a quick test)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override BATCH_SIZE (per GPU)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override LR for every variant (FAMILY_LR is ignored when set)')
    parser.add_argument('--patience', type=int, default=None, help='Override EARLY_STOP_PATIENCE')
    parser.add_argument('--cpu-offload', type=str, default=None, choices=['auto', 'on', 'off'],
                        help='Override FSDP_CPU_OFFLOAD (auto = only for models that do not fit)')
    parser.add_argument('--deterministic', action='store_true',
                        help='cudnn deterministic algorithms (exact reruns, slower)')
    parser.add_argument('--target-precision', type=float, default=None,
                        help=f'Override TARGET_PRECISION for Perfect (default {c.TARGET_PRECISION})')
    parser.add_argument('--selection-metric', type=str, default=None,
                        help=f"Override SELECTION_METRIC (default '{c.SELECTION_METRIC}'), "
                             f"e.g. val_selection_score, val_ap, val_f1opt_f1_perfect")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.patience is not None:
        overrides['early_stop_patience'] = args.patience
    if args.cpu_offload is not None:
        overrides['fsdp_cpu_offload'] = {'auto': 'auto', 'on': True, 'off': False}[args.cpu_offload]
    if args.deterministic:
        overrides['deterministic'] = True
    if args.target_precision is not None:
        if not 0.0 < args.target_precision <= 1.0:
            parser.error('--target-precision must be in (0, 1]')
        overrides['target_precision'] = args.target_precision
    if args.selection_metric is not None:
        if not args.selection_metric.startswith('val_'):
            parser.error("--selection-metric must start with 'val_'")
        overrides['selection_metric'] = args.selection_metric

    if torch.cuda.is_available():
        n_gpus = min(args.num_gpus, torch.cuda.device_count())
        devices = list(range(max(n_gpus, 1)))
    else:
        devices = ['cpu']

    dataset_dir = c.DATASETS[args.dataset]

    print("=" * 60)
    print(f"🚀 experiment='{args.experiment}' dataset='{args.dataset}' model='{args.model}' "
          f"devices={devices} overrides={overrides or 'none'}")
    print("=" * 60)

    clf = WormClassificationModel(model_name=args.model, experiment_name=args.experiment, **overrides)
    results = clf.train(dataset_dir=dataset_dir, dataset_name=args.dataset, devices=devices)

    if results:  # empty {} on non-rank-0 workers under torchrun
        out_path = Path(c.OUTPUT_DIR) / args.experiment / 'train_results' / args.dataset / f'{args.model}_job_summary.json'
        atomic_json_dump(results, out_path)
        print(f"\n🎉 Finished {args.model} on {args.dataset}. Summary: {out_path}")
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()