import torch.distributed as dist

import config as c
from worm_mrcnn import WormMRCNN


def run_one_dataset(dataset_choice, run_id, resume_from=None):
    model = WormMRCNN(dataset_choice=dataset_choice, run_id=run_id)

    model.log_info(f"\n{'#' * 70}\n# [DATASET: {dataset_choice}] starting\n{'#' * 70}\n")

    model.log_info(f"[DATASET: {dataset_choice}] training...")
    _, best_ckpt, run_checkpoint_dir, tb_log_dir = model.train(resume_from=resume_from)
    model.log_info(f"[DATASET: {dataset_choice}] training complete. best checkpoint: {best_ckpt}")

    if c.RUN_TEST_AFTER_TRAIN and best_ckpt is not None:
        if model.rank == 0:
            model.log_info(f"[DATASET: {dataset_choice}] running evaluation on test split...")
            results = model.test(checkpoint_path=best_ckpt)
            print(f"[DATASET: {dataset_choice}] test results: {results['summary']}", flush=True)
        if model.is_distributed:
            dist.barrier()

    if model.rank == 0 and best_ckpt is not None:
        onnx_path = model.export_onnx(checkpoint_path=best_ckpt)
        model.log_info(f"[DATASET: {dataset_choice}] exported to ONNX: {onnx_path}")

    model.log_info(f"[DATASET: {dataset_choice}] done.\n")
    return model


def main():
    # a shared run_id keeps every dataset's checkpoints/tensorboard logs under one run folder
    probe = WormMRCNN(dataset_choice=c.DATASETS_TO_RUN[0])
    run_id = probe.run_id

    probe.log_info(f"[Info] device = {probe.device}")
    probe.log_info(f"[Info] distributed = {probe.is_distributed} | world_size = {probe.world_size}")
    probe.log_info(f"[Info] Experiment Run ID = {run_id}")

    resume_from = c.RESUME_FROM if len(c.DATASETS_TO_RUN) == 1 else None

    for i, dataset_choice in enumerate(c.DATASETS_TO_RUN, start=1):
        probe.log_info(f"[Info] processing dataset {i}/{len(c.DATASETS_TO_RUN)}: {dataset_choice}")
        run_one_dataset(dataset_choice, run_id=run_id, resume_from=resume_from)

    probe.log_info("\n[Info] all datasets finished.")

    if probe.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()