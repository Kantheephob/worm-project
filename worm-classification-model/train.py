import json
import argparse
from pathlib import Path

import torch

from classifier import WormClassificationModel
from common import atomic_json_dump
import config as c


def main():
    parser = argparse.ArgumentParser(description='Train one dataset x one model (or family) on LANTA')
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
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        overrides['lr'] = args.lr

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