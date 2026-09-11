import json
from pathlib import Path
import argparse

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
    args = parser.parse_args()

    dataset_dir = c.DATASETS[args.dataset]

    print("=" * 60)
    print(f"🚀 experiment='{args.experiment}' dataset='{args.dataset}' model='{args.model}'")
    print("=" * 60)

    clf = WormClassificationModel(model_name=args.model, experiment_name=args.experiment)
    results = clf.train(dataset_dir=dataset_dir, dataset_name=args.dataset, devices=list(range(c.NUM_GPUS)))

    if results:  # empty {} on non-rank-0 workers under torchrun
        out_path = Path(c.OUTPUT_DIR) / args.experiment / 'train_results' / args.dataset / f'{args.model}_job_summary.json'
        atomic_json_dump(results, out_path)
        print(f"\n🎉 Finished {args.model} on {args.dataset}. Summary: {out_path}")
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()